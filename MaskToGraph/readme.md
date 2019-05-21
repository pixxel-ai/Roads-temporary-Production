# Convert Masks to Graphs

Input :
1. Path to folder containing all masks
2. Path to `output file` without extension

Output:
Graphs for each mask in LineString format in a file called `output file.txt`

You can read LineString objects using `Shapely`

The file `Plot_line_strings.ipynb` can be referred to understand how the LineString objects can be read and visualized

you can use this script like:
`python3 mask_to_graph_script.py "/path/to/masks/" "output_file_without_extension"`