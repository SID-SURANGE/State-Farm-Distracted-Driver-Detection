# State-Farm-Distracted-Driver-Detection

<p><i> Kaggle hosted the challenge few years ago which focused on identifying distracted drivers using Computer Vision <br>
    Details of challenge can be found here - https://www.kaggle.com/c/state-farm-distracted-driver-detection </i></p>
    <hr>
    
<h3> Problem Description </h3>
<p>State Farm launched a kaggle competition few years ago called <b>“State Farm Distracted Driver Detection”</b>, where given driver images, each taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc). The goal was to predict the likelihood of what the driver is doing in each picture.</p>
    
  <h3>Dataset details -</h3> 
  <ul>
  <li> Image Size - 480 X 640 pixels</li>
  <li> Training Images count - 22424 images </li>
  <li> Test Images count - 79726 images </li>
  <li> Image type - RGB </li>
  <li> Image field of view - Dashboard images with view of Driver and passenger </li>
  <li> The 10 classes to predict are: <br>
        <ul>
          <li>    c0: safe driving<br>
          <li>    c1: texting - right<br>
          <li>    c2: talking on the phone - right<br>
          <li>    c3: texting - left<br>
          <li>    c4: talking on the phone - left<br>
          <li>    c5: operating the radio<br>
          <li>    c6: drinking<br>
          <li>    c7: reaching behind<br>
          <li>    c8: hair and makeup<br>
          <li>    c9: talking to passenger</ul>
   <li> Loss - multi-class logarithmic loss</li>
  </ul>
  
  <h3> Impementation Details</h3>
  <ul>
  <li> DL Model - CNN's build from scratch ( 6 Conv Layer, 5 Dropout Layer, 3 Dense Layer)
  <li> Framework - Keras / Pytorch version in the process.
  <li> CNN Model Visualization - GradCAM
  <li> Final Accuracy -Train acc - 99.06%, Val acc-99 .46%
  </ul>

 
<h3> GRAD-CAM implementation for a test image with label drinking </h3>
<p align='center'> 
<img src="superimposed_img.jpg"
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px;"
     height=400
     width=450/>
</p>

<h3> Sample prediction over test image using Flask </h3>
<p align='center'> 
<img src="Sample_prediction.JPG"
     alt="Markdown Monster icon"
     style="float: center; margin-right: 10px;"
     height=500
     width=500 />
</p>
