import warnings
import time

from copy import deepcopy

import numpy as np

from modeling.utils.sample_generator_utils import offset_pixel_coords
from modeling.Sample import Sample, View
from modeling.utils.alignment_utils import reconstruct_adjustments_from_unadjusted_adjusted_pairs

from modeling.constants import (
    SAMPLE_METADATA_ATTEMPTS,
    SAMPLE_METADATA_EXCEPTIONS,
    POLYGON_COUNT_PREFIX,
    PIXEL_COUNT_PREFIX,
    SAMPLE_GENERATION_TIMING_PREFIX
)

class WindowedDatasetAdaptor:
    def __init__(
        self,
        orthomosaics,
        label_map,
        sample_location_generation_strategy,
        keypoint_conversion_strategy,
        mask_generation_strategy,
        tile_x=2048,
        tile_y=2048,
        mask_x=2048,
        mask_y=2048,
        backend="auto"
    ):
        self.orthomosaics = orthomosaics
        self.mask_x = mask_x
        self.mask_y = mask_y
        self.tile_x = tile_x
        self.tile_y = tile_y
        self._backend = backend
        self.label_map = label_map

        self.sample_location_generation_strategy = sample_location_generation_strategy
        self.keypoint_conversion_strategy = keypoint_conversion_strategy
        self.mask_generation_strategy = mask_generation_strategy

        self.sample_location_generation_strategy.initializeLocationGenerationStrategy(self.tile_x, self.tile_y)
        self.mask_generation_strategy.initialize_masking_strategy(self.mask_x, self.mask_y, self.tile_x, self.tile_y)

        warnings.filterwarnings("ignore")

    def __len__(self) -> int:
        return len(self.sample_location_generation_strategy)

    def generate_sample(self, index):
        #Record the time at which sample generation started
        t_select_ortho = time.time()

        #Sample a location that we will present
        sample_location = self.sample_location_generation_strategy.getSampleLocation(index)
        exceptions = sample_location.getGenerationMetadata().getExceptions()

        #Record the time at which we have our sample location
        t_bbox_choice = time.time()

        #Get the IDs of the spatial objects in the sample
        building_ids = [b.getId() for b in sample_location.getBuildings()]
        road_line_ids = [rl.getId() for rl in sample_location.getRoadLines()]

        #Get the orthomosaic that we will return, and so we can get the details of the sample
        orthomosaic = self.orthomosaics[sample_location.getOrthomosaicIdx()]

        #Get the location of the sample
        x_p = sample_location.getX()
        y_p = sample_location.getY()

        #Get the adjusted, and unadjusted pairs for both road lines and building polygons
        adjusted_buildings = orthomosaic.get_buildings(ids=building_ids, adjusted=True)
        adjusted_roadlines = orthomosaic.get_road_lines(ids=road_line_ids, adjusted=True)
        unadjusted_buildings = sample_location.getBuildings()
        unadjusted_roadlines = sample_location.getRoadLines()

        #Offset the buildings so they are now in the coordinates of the image
        unadjusted_buildings_copy = deepcopy(unadjusted_buildings)
        adjusted_buildings_copy = deepcopy(adjusted_buildings)
        for unadjusted_building, adjusted_building in zip(unadjusted_buildings_copy, adjusted_buildings_copy):
            unadjusted_building.setGeometry(offset_pixel_coords(unadjusted_building.getGeometry("pixels"), x_p, y_p), "pixels")
            adjusted_building.setGeometry(offset_pixel_coords(adjusted_building.getGeometry("pixels"), x_p, y_p), "pixels")
        building_adjustments = reconstruct_adjustments_from_unadjusted_adjusted_pairs(unadjusted_buildings_copy, adjusted_buildings_copy, x_p, y_p)

        #Offset the road lines so they are now in the coordinates of the image
        unadjusted_roadlines_copy = deepcopy(unadjusted_roadlines)
        adjusted_roadlines_copy = deepcopy(adjusted_roadlines)
        for unadjusted_roadline, adjusted_roadline in zip(unadjusted_roadlines_copy, adjusted_roadlines_copy):
            
            unadjusted_roadline.setGeometry(offset_pixel_coords(unadjusted_roadline.getGeometry("pixels"), x_p, y_p), "pixels")
            for unadjusted_labeled_sub_road_line in unadjusted_roadline.get_labeled_sub_lines():
                unadjusted_labeled_sub_road_line.setGeometry(offset_pixel_coords(unadjusted_labeled_sub_road_line.getGeometry("pixels"), x_p, y_p), "pixels")
            
            adjusted_roadline.setGeometry(offset_pixel_coords(adjusted_roadline.getGeometry("pixels"), x_p, y_p), "pixels")
        
        roadline_adjustments = reconstruct_adjustments_from_unadjusted_adjusted_pairs(unadjusted_roadlines_copy, adjusted_roadlines_copy, x_p, y_p)

        #Combine the adjustments into the resulting vector field that will be passed with the sample
        adjustments = building_adjustments + roadline_adjustments

        #Record the time at which data loading started
        t_load_data = time.time()

        #Read the data from the orthomosaic
        try:
            color_data = orthomosaic.read(x_p, y_p, sample_location.getXDim(), sample_location.getYDim(), center_xy=False)
        # If an exception occurs, then make the sample blank
        # pylint: disable-next=broad-exception-caught
        except Exception as e:
            color_data = np.zeros((sample_location.getXDim(), sample_location.getYDim(), 3))
            adjustments = []
            unadjusted_buildings_copy = []
            unadjusted_roadlines_copy = []
            if str(type(e)) in exceptions.keys():
                exceptions[str(type(e))] += 1
            else:
                print("Untracked exception occured in Dataset Adaptor. Returning Empty Sample...", e)

        # Log the details of the attempt to generate a sample
        t_compute_telemetry = time.time()
        sample_metadata = {}
        sample_metadata[SAMPLE_METADATA_EXCEPTIONS] = exceptions
        sample_metadata[SAMPLE_METADATA_ATTEMPTS] = sample_location.getGenerationMetadata().getAttempts()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "select_ortho_time"] = t_bbox_choice - t_select_ortho
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "generate_sample_point_time"] = sample_location.getGenerationMetadata().getGenerationSec()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "get_annotation_time"] = sample_location.getGenerationMetadata().getAnnotationSec()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "validate_sample_time"] = sample_location.getGenerationMetadata().getValidationSec()
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "select_window_time"] = t_load_data - t_bbox_choice
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "load_pixels_time"] = t_compute_telemetry - t_load_data

        for label in self.label_map.getAllLabels():
            
            sample_metadata[POLYGON_COUNT_PREFIX + label] = sum(1 if b.getLabel() == label else 0 for b in unadjusted_buildings_copy)
            sample_metadata[PIXEL_COUNT_PREFIX + label] = sum(b.getGeometry("pixels").area if b.getLabel() == label else 0 for b in unadjusted_buildings_copy)
            
            for roadline in unadjusted_roadlines_copy:
                    sample_metadata[POLYGON_COUNT_PREFIX + label] += sum(1 if b.getLabel() == label else 0 for b in roadline.get_labeled_sub_lines())
                    sample_metadata[PIXEL_COUNT_PREFIX + label] += sum(b.getGeometry("pixels").length if b.getLabel() == label else 0 for b in roadline.get_labeled_sub_lines())
            
        sample_metadata[SAMPLE_GENERATION_TIMING_PREFIX + "compute_telemetry_time"] = time.time() - t_compute_telemetry

        v = View(raw_imagery=color_data, adjustments=adjustments, orthomosaic=orthomosaic)
        return Sample(x=x_p, y=y_p, views=[v], buildings=unadjusted_buildings_copy, road_lines=unadjusted_roadlines_copy, metadata=sample_metadata, label_map=self.label_map)

    def set_backend(self, backend):
        for ortho in self.orthomosaics:
            ortho.set_backend(backend)

    def get_backend(self):
        return self._backend
