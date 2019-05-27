# Temporary Production for roads models

This package contains helper code that enerates masks and graphs for each image in a given folder using a trained Fastai model.

`generate_predictions.py` can be used as follows : <br>
`python3 generate_predictions.py model_path in_folder out_folder model_imsize graphs_output resize_to` <br>  
```
   Parameters:
  model_path : Path of the Fastai model file
   in_folder : Path containing images to predict on
  out_folder : Path to output masks to
      model_imsize : Model Image Size (not output image size)
graphs_output: Path to file (without extension) to which graphs will be output
   resize_to : [OPTIONAL] Output image size of masks and graphs

      Outputs: Predicted masks for each image in `out_folder`
               Graphs generated for each mask in `graphs_output.txt`
```

`logs.log` will record operations for each run. The fill be appended and not rewritten after each run <br>
It will contain :
1. Arguments passed
2. `Success` or `Failure` Flags for each image processed
3. Each log will contain time-stamp information (GMT -- NOT IST)