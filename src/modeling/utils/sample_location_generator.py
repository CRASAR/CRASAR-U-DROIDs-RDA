import time
from multiprocessing import Pool

import numpy as np

from modeling.constants import TRACKED_EXCEPTIONS
from modeling.utils.sample_generator_utils import generate_sample_point, get_valid_lines, get_valid_buildings
from modeling.utils.building_frame_generation import get_candidate_samples_center
from modeling.utils.sample_presentation import RealTimeSampleLocationPresentationStrategy, PregeneratedSampleLocationPresentationStrategy
from modeling.utils.random_utils import reseed_distributed
from modeling.Spatial import MultiLabeledRoadLineFactory

# This is a class that implements a strategy for generating sample locations in an orthomosaic
# The expectation is that if you want to get different behavior from your data generator, you
# Can pass these different strategies to your code and it it will utilize the different strategy
# as needed. This is the entry point into the class hierarchy.
class SampleLocationGenerationStrategy:
    def __init__(self, strategy_name, sample_location_presentation_strategy, annotator):
        self._strategy_name = strategy_name
        self._sample_location_presentation_strategy = sample_location_presentation_strategy
        self._annotator = annotator
        self._xdim = None
        self._ydim = None
    def initializeLocationGenerationStrategy(self, xdim, ydim):
        self._xdim = int(xdim)
        self._ydim = int(ydim)
    def getStrategyName(self):
        return self._strategy_name
    def getAnnotator(self):
        return self._annotator
    def getSampleLocation(self, index):
        raise NotImplementedError("getSampleLocations must be implemented by a subclass")
    def __len__(self):
        return len(self._sample_location_presentation_strategy)

# This is a class that is specifically for sample generation strategies that involve generating
# Samples in real time when the function getSampleLocation is called.
class RealTimeSampleLocationGenerationStrategy(SampleLocationGenerationStrategy):
    def __init__(self, name, orthomosaics, annotator, sample_location_presentation_strategy):
        if not isinstance(sample_location_presentation_strategy, RealTimeSampleLocationPresentationStrategy):
            raise ValueError("sample_location_presentation_strategy must be an instance of ",
                             RealTimeSampleLocationPresentationStrategy,
                             "instead found",
                             type(sample_location_presentation_strategy))
        super().__init__(name, sample_location_presentation_strategy, annotator)

        self._orthomosaics = orthomosaics
    def getSampleLocation(self, index):
        raise NotImplementedError("getSampleLocations must be implemented by a subclass")

# This is a class that iteratively generates a random sample location until it finds a valid one
class RandomSampleLocationGenerationStrategy(RealTimeSampleLocationGenerationStrategy):
    def __init__(self, orthomosaics, annotator, sample_location_presentation_strategy, sample_acceptance_persistence=25, seed_range=10e6):
        super().__init__("random", orthomosaics, annotator, sample_location_presentation_strategy)
        self._sample_acceptance_persistence = int(sample_acceptance_persistence)
        self.__rs = np.random.RandomState()
        self.__seed_range = int(float(seed_range))
        self._samples_generated = 0

    def getSampleLocation(self, index):
        reseed_distributed(self._samples_generated, self.__rs, self.__seed_range)
        self._samples_generated += 1

        orthomosaic_idx = self.__rs.randint(0, len(self._orthomosaics))
        ortho = self._orthomosaics[orthomosaic_idx]

        attempts = 0
        t_sample_generation = 0
        t_sample_annotation = 0
        t_sample_validation = 0
        accepted = False
        exceptions = {e:0 for e in TRACKED_EXCEPTIONS}
        while((not accepted) and attempts < self._sample_acceptance_persistence):
            t_0 = time.time()
            x_p, y_p = generate_sample_point(ortho, self.__rs)
            t_1 = time.time()

            validation_call_args = self._annotator.make_sample_annotation_call_args(x_p, y_p, self._xdim, self._ydim, ortho, orthomosaic_idx, None)
            sample_candidate = self._annotator.annotate_sample(*validation_call_args)

            #Combine the exceptions with what we have so far
            exceptions = {e:exceptions[e] + sample_candidate.getGenerationMetadata().getExceptions()[e] for e in exceptions.keys()}

            t_2 = time.time()
            if len(sample_candidate.getBuildings()) > 0 or len(sample_candidate.getRoadLines()) > 0:
                accepted = True
            t_3 = time.time()

            t_sample_generation += t_1-t_0
            t_sample_annotation += t_2-t_1
            t_sample_validation += t_3-t_2

            attempts += 1

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(attempts, t_sample_generation, t_sample_annotation, t_sample_validation, exceptions)

        # Return the valid road lines that were generated for the sample.
        result = SampleLocation(x=x_p,
                                y=y_p,
                                x_dim=self._xdim,
                                y_dim=self._ydim,
                                buildings=sample_candidate.getBuildings(),
                                roadlines=sample_candidate.getRoadLines(),
                                orthomosaic_idx=orthomosaic_idx,
                                generation_meta=generation_meta)

        self._sample_location_presentation_strategy.observeSampleLocation(result)
        return self._sample_location_presentation_strategy.getSampleLocation(index)


# This is a subclass that generates sample locations all at once and validates them for the user
# so that when the user calls the getSampleLocation function, there is a valid sample waiting for
# them to pass to a model.
class PregeneratedSampleLocationGenerationStrategy(SampleLocationGenerationStrategy):
    def __init__(self, name, orthomosaics, annotator, sample_location_presentation_strategy, sample_generator_process_pool_size=6):
        if not isinstance(sample_location_presentation_strategy, PregeneratedSampleLocationPresentationStrategy):
            raise ValueError("sample_location_presentation_strategy must be an instance of ",
                             PregeneratedSampleLocationPresentationStrategy,
                             "instead found",
                             type(sample_location_presentation_strategy))
        super().__init__(name, sample_location_presentation_strategy, annotator)
        self._sample_generator_process_pool_size = sample_generator_process_pool_size
        self._annotator = annotator
        self._samples = []
        self._orthomosaics = orthomosaics

    def initializeLocationGenerationStrategy(self, xdim, ydim):
        super().initializeLocationGenerationStrategy(xdim, ydim)
        self._samples = self._pregenerate_sample_locations(self._orthomosaics)
        self._sample_location_presentation_strategy.initialize_samples(self._samples)

    def getSampleLocation(self, index):
        return self._sample_location_presentation_strategy.getSampleLocation(index)

    def _pregenerate_sample_locations(self, orthomosaics):
        testing_locations = []
        for i, orthomosaic in enumerate(orthomosaics):
            testing_locations.extend(self._get_sample_locations_to_validate(orthomosaic, i))

        result = []
        with Pool(processes=self._sample_generator_process_pool_size) as pool:
            candidate_samples = pool.starmap(self._annotator.annotate_sample, testing_locations)

        for candidate_sample in candidate_samples:
            if len(candidate_sample.getBuildings()) > 0 or len(candidate_sample.getRoadLines()) > 0:
                result.append(candidate_sample)
        return result

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        raise NotImplementedError("_get_sample_locations_to_validate must be implemented by a subclass")

# This is a subclass that generates samples for building damage assessment and alignment training that
# Attempts to include as many buildings in a frame as possible while keeping them within the range
# of adjustment_buffer_distance_px pixels from the edge of the frame
class CenteredBuildingSampleStrategy(PregeneratedSampleLocationGenerationStrategy):
    def __init__(self, adjustment_buffer_distance_px, annotator, sample_location_presentation_strategy, orthomosaics, sample_generator_process_pool_size=6):
        if not isinstance(annotator, BDASampleAnnotator):
            raise ValueError("CenteredSampleStrategy is only defined for samples that can be validated using the BDASampleAnnotator.")

        self._adjustment_buffer_distance_px = int(adjustment_buffer_distance_px)
        super().__init__("Centered", orthomosaics, annotator, sample_location_presentation_strategy, sample_generator_process_pool_size)

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        bda_sample_validation_calls = []
        frames_of_buildings = get_candidate_samples_center(orthomosaic,
                                                           self._xdim,
                                                           self._ydim,
                                                           self._adjustment_buffer_distance_px,
                                                           adjusted=self._annotator.generatesAdjustedSamples())
        for frame_of_buildings in frames_of_buildings:
            x = frame_of_buildings[0].centroid.x - self._xdim/2
            y = frame_of_buildings[0].centroid.y - self._ydim/2
            building_ids = frame_of_buildings[1]
            bda_sample_validation_calls.append(self._annotator.make_sample_annotation_call_args(x,
                                                                                                y,
                                                                                                self._xdim,
                                                                                                self._ydim,
                                                                                                orthomosaic,
                                                                                                orthomosaic_idx,
                                                                                                building_ids))
        return bda_sample_validation_calls

# This is a subclass that generates sample for building and road damage assessment by uniformly tiling
# the image into a grid.
class GridSampleStrategy(PregeneratedSampleLocationGenerationStrategy):
    def __init__(self, adjustment_buffer_distance_px, annotator, sample_location_presentation_strategy, orthomosaics, sample_generator_process_pool_size=6):
        self._adjustment_buffer_distance_px = adjustment_buffer_distance_px
        super().__init__("Grid", orthomosaics, annotator, sample_location_presentation_strategy, sample_generator_process_pool_size)

    def _get_sample_locations_to_validate(self, orthomosaic, orthomosaic_idx):
        # Create a list to store the candidate samples
        sample_validation_calls = []

        # Iterate over the orthomosaic looking for buildings that need to be labeled
        for x in np.arange(0-self._adjustment_buffer_distance_px, orthomosaic.get_width(), self._xdim-2*self._adjustment_buffer_distance_px):
            for y in np.arange(0-self._adjustment_buffer_distance_px, orthomosaic.get_height(), self._ydim-2*self._adjustment_buffer_distance_px):
                call = self._annotator.make_sample_annotation_call_args(x, y, self._xdim, self._ydim, orthomosaic, orthomosaic_idx, None)
                sample_validation_calls.append(call)
        return sample_validation_calls

class SampleLocation:
    def __init__(self, x, y, x_dim, y_dim, buildings, roadlines, orthomosaic_idx, generation_meta=None):
        self._x = x
        self._y = y
        self._x_dim = x_dim
        self._y_dim = y_dim
        self._buildings = buildings
        self._roadlines = roadlines
        self._orthomosaic_idx = orthomosaic_idx
        self._generation_meta = generation_meta
        if self._generation_meta is None:
            self._generation_meta = SampleLocationGenerationMetadata()

    def getX(self):
        return self._x
    def getY(self):
        return self._y
    def getXDim(self):
        return self._x_dim
    def getYDim(self):
        return self._y_dim
    def getBuildings(self):
        return self._buildings
    def getRoadLines(self):
        return self._roadlines
    def getOrthomosaicIdx(self):
        return self._orthomosaic_idx
    def getGenerationMetadata(self):
        return self._generation_meta

class SampleLocationGenerationMetadata:
    def __init__(self, attempts=1, generation_sec=0.0, annotation_sec=0.0, validation_sec=0.0, exceptions=None):
        self._exceptions = exceptions
        if self._exceptions is None:
            self._exceptions = {}
        self._attempts = attempts
        self._generation_sec = generation_sec
        self._annotation_sec = annotation_sec
        self._validation_sec = validation_sec
    def getExceptions(self):
        return self._exceptions
    def getAttempts(self):
        return self._attempts
    def getGenerationSec(self):
        return self._generation_sec
    def getAnnotationSec(self):
        return self._annotation_sec
    def getValidationSec(self):
        return self._validation_sec

class SamplePregenerationAnnotator:
    def __init__(self, generate_adjusted_samples, center_xy=False):
        self._generate_adjusted_samples = generate_adjusted_samples
        self._center_xy = center_xy
    def expectsCenteredXY(self):
        return self._center_xy
    def generatesAdjustedSamples(self):
        return self._generate_adjusted_samples

class BDASampleAnnotator(SamplePregenerationAnnotator):
    def __init__(self, generate_adjusted_samples, center_xy=False, building_intersection_proportion_threshold=0.0):
        super().__init__(generate_adjusted_samples, center_xy)
        self._building_intersection_proportion_threshold = float(building_intersection_proportion_threshold)
    def make_sample_annotation_call_args(self, x, y, x_dim, y_dim, orthomosaic, orthomosaic_idx, ids=None):
        return [x,
                y,
                x_dim,
                y_dim,
                orthomosaic.get_buildings(adjusted=self._generate_adjusted_samples, ids=ids),
                orthomosaic_idx]
    def annotate_sample(self, x, y, x_dim, y_dim, buildings, orthomosaic_idx):
        t0 = time.time()
        # Get the valid polygons for this window
        exceptions = {e:0 for e in TRACKED_EXCEPTIONS}
        valid_buildings, monitored_exceptions = get_valid_buildings(
            x=x,
            y=y,
            buildings=buildings,
            x_dim=x_dim,
            y_dim=y_dim,
            building_intersection_proportion_threshold=self._building_intersection_proportion_threshold,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        t1 = time.time()

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, monitored_exceptions)

        # Return the valid buildings that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=valid_buildings,
                              roadlines=[],
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)

class RDASampleAnnotator(SamplePregenerationAnnotator):
    def make_sample_annotation_call_args(self, x, y, x_dim, y_dim, orthomosaic, orthomosaic_idx, _):
        return [x,
                y,
                x_dim,
                y_dim,
                orthomosaic.get_road_lines(adjusted=self._generate_adjusted_samples),
                orthomosaic.get_road_line_annotation_polygons(),
                orthomosaic_idx]

    def annotate_sample(self, x, y, x_dim, y_dim, roadlines, annotation_polygons, orthomosaic_idx):
        t0 = time.time()
        exceptions = {e:0 for e in TRACKED_EXCEPTIONS}
        valid_road_lines, monitored_exceptions = get_valid_lines(
            x,
            y,
            roadlines,
            x_dim=x_dim,
            y_dim=y_dim,
            exceptions_to_track=exceptions,
            center_xy=self._center_xy,
        )
        labeled_road_lines = MultiLabeledRoadLineFactory(valid_road_lines, annotation_polygons)
        t1 = time.time()

        # Store the metadata associated with the attempt to generate a sample.
        generation_meta = SampleLocationGenerationMetadata(1, 0, 0, t1-t0, monitored_exceptions)

        # Return the valid road lines that were generated for the sample.
        return SampleLocation(x=x,
                              y=y,
                              x_dim=x_dim,
                              y_dim=y_dim,
                              buildings=[],
                              roadlines=labeled_road_lines,
                              orthomosaic_idx=orthomosaic_idx,
                              generation_meta=generation_meta)
