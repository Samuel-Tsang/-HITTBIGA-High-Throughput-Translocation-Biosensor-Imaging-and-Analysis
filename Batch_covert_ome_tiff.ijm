// To batch stack individual images (e.g. channels, timepoints) of the same position and save it as .ome.tiff
// images of each position has to be in separate folders (i.e. export from CellProfiler)


input = getDirectory("Choose input Directory"); // define the directory containing all the image folders
folders = getFileList(input); // get a list of all the folders
output = getDirectory("Choose output Directory"); // define the directory to save all the .ome.tiff hyperstacks
experiment_name = getString("Input experiment name of the exported files", "Experiment"); // CHANGE EXPERIMENT NAME IF NECESSARY
output_path = output + experiment_name + "_s"; //path and file name of the edited .ome.tiff
extension = ".ome.tiff"; // extension of the edited .ome.tiff


setBatchMode(true);


// To work with individual image folder one by one
for (i=0; i<folders.length; i++)
{
	working_folder = input + folders[i];
	file_list = getFileList(working_folder);
	file = file_list[0]; // the first image file in the working folder
	working_file = working_folder + file;
	position = replace(folders[i], "/","");
	run("Bio-Formats Importer", "open=["+working_file+"] color_mode=Default group_files rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT"); // use bio-formats plugin to import the first image, automatically recoginze all the images with the same naming pattern and stack them together
	run("Bio-Formats Exporter", "save=["+output_path + position + extension+"] compression=Uncompressed"); // use bio-formats plugin to export the stack images as .ome.tiff to a defined directory
	print("s" + position + "_done_" + getTime());
	run("Close All");
	// run("Collect Garbage"); 
}


setBatchMode(false);


print("All done_" + i + "_positions processed");