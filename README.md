# Demo-water-segmentation
**This is our team project for the MultiDisciplinary Project in HCMUT.**

## License
This project is forked from my leader repository [https://github.com/ledong0110/Demo-water-segmentation](https://github.com/ledong0110/Demo-water-segmentation).

## Description
### Why called MultiDisciplinary Project
This project is combine the **Power of Conputer Engineering** and the **Wisdom of Computer Science** :
* Computer Science team will think off how to use the model, about  data structures and solution to build the model - Software
* My friend in Conputer Engineering Faculty toke responsibility on IOU devices - Hardware

### What is this project about
The purpose of this project is building an automatic system begin with Taking pictures at some points in Ho CHi Minh city, then Using AI model to detect the Flood and its Depth. Then through a mobile application, users will recieved notification about the flooding areas so that they can avoid them.

<img src="https://kaze.com.vn/newsmultidata/blog11.jpg" width="500" alt="Screenshot of My Project">

## Detail Work
1. We using Detection - Segmentation model for Segmenting Flood Area. See the picture below
<img src="https://github.com/WinerDeCoder/Demo-water-segmentation/assets/136697023/13e6f96e-cacf-4ab7-bd75-dd29e4ce5600" width="700" alt="Screenshot of My Project">

2. There is no way we can predict the depth of water only using model => I use Heuristic method
   - When setting camera in an area, we will choose a fixed line, could be any line in that frame ( this will be refer line ) . Then we calculate its height to the street. We also calculate that length in the frame of camera ( we could do this because every thing is fixed )
   - Next, when flood is detected and segmented, we will project the line down when touch the segmentation mask => We will know how far are they ==> Then using the fixed line, we can calculate the real depth just by some simple math. See the picture below
<img src="https://github.com/WinerDeCoder/Demo-water-segmentation/assets/136697023/283203d1-8eb2-47bd-b033-67f6cb000ee3" width="700" alt="Screenshot of My Project">

## Other Repository for this Project
* [https://github.com/ledong0110/Multidisciplinary-Project](https://github.com/ledong0110/Multidisciplinary-Project)
* [https://github.com/ledong0110/multi-project-app](https://github.com/ledong0110/multi-project-app)
* [https://github.com/ledong0110/Multidisciplinary-Project](https://github.com/ledong0110/Multidisciplinary-Project)

> [!NOTE]
> Read our report for more detail .pdf

   
