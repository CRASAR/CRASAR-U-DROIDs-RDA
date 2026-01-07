def find_geotif_file_prefix_match(target_geotiff, candidate_files):
    a_c = target_geotiff.split(".tif")[0]
    for b in candidate_files:
        if(a_c.lower() == b.split(".tif")[0].lower()):
            return b
    return None