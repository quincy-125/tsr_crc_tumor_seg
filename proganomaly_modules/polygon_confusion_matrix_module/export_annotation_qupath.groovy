// This script will export the annotations of each slide in the currently-open project
// to a GeoJSON file named after the slide's name.
// If extends an example from the QuPath documentation at https://qupath.readthedocs.io/en/stable/docs/scripting/overview.html.
// It can be run within QuPath by clicking Automate -> Show script editor, and run by clicking Run -> Run within the script editor.
// This script will export the annotations of each slide in the currently-open project
// to a GeoJSON file named after the slide's name.
// If extends an example from the QuPath documentation at https://qupath.readthedocs.io/en/stable/docs/scripting/overview.html.

// It can be run within QuPath by clicking Automate -> Show script editor, and run by clicking Run -> Run within the script editor.

// Get total image count:
def project = getProject()
boolean prettyPrint = false
        // Load a JSON parser (see 
        // https://qupath.readthedocs.io/en/stable/docs/scripting/overview.html#serialization-json )
        def gson = GsonTools.getInstance(prettyPrint)
        // Loop through each image in the open project, and export its annotations:
        for (entry in project.getImageList()) {
            // Read all annotations on the slide:
            def hierarchy = entry.readHierarchy()
    def annotations = hierarchy.getAnnotationObjects() 
            def path = "C:/path/to/desired/directory/for/geojson/files/" + entry + ".geojson"
             // This follows the example at 
             // https://qupath.readthedocs.io/en/stable/docs/advanced/exporting_annotations.html#geojson
            // As noted at the link above, 
            // 'FEATURE_COLLECTION' is a standard GeoJSON format for multiple objects
        exportObjectsToGeoJson(annotations, path, "FEATURE_COLLECTION")
}