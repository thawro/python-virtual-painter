# AI-based Virtual Painter
Hand landmarks are found using DL model and images are processed with openCV.

# Virtual Painter app

## Actions
* `DRAW` - thumb tip and middle finger tip very close to each other
* `SIZE CHANGE` - brush size changing is started when thumb tip and index finger tip are very close to each other and the size is controlled by the distance between those tips
* `COLOR PICK` - while not `DRAW` move index finger tip over one of the top colors 
* `COLOR PALETTE` - connect thumb tip and middle finger tip while index finger tip is over the color that you would like to change
* `ERASER` - same as `COLOR PICK`, but indef finger tip must go over the eraser icon

## Examples for different hand landmark models 
### Mediapipe [hand landmarks model](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) 
https://github.com/thawro/python-virtual-painter/assets/50373360/effafd7e-c522-4a6a-a217-294d81a64669



## Self trained hand landmarks model [TODO]
