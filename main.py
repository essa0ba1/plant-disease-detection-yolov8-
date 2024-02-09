from ultralytics import YOLO  
import streamlit as st 
import plotly.graph_objs as go
from PIL import Image,ImageDraw,ImageFilter
import numpy as np 
from io import BytesIO
colors = {
    'Apple Scab Leaf': ['rgb(165, 42, 42)', (165, 42, 42)],  # Brown
    'Apple leaf': ['rgb(128, 0, 128)', (128, 0, 128)],  # Purple
    'Apple rust leaf': ['rgb(0, 255, 0)', (0, 255, 0)],  # Green
    'Bell_pepper leaf spot': ['rgb(255, 215, 0)', (255, 215, 0)],  # Gold
    'Bell_pepper leaf': ['rgb(139, 69, 19)', (139, 69, 19)],  # Brown
    'Blueberry leaf': ['rgb(128, 128, 128)', (128, 128, 128)],  # Gray
    'Cherry leaf': ['rgb(0, 128, 0)', (0, 128, 0)],  # Dark green
    'Corn Gray leaf spot': ['rgb(255, 0, 0)', (255, 0, 0)],  # Red
    'Corn leaf blight': ['rgb(255, 165, 0)', (255, 165, 0)],  # Orange
    'Peach leaf': ['rgb(255, 20, 147)', (255, 20, 147)],  # Pink
    'Potato leaf early blight': ['rgb(255, 105, 180)', (255, 105, 180)],  # Hot pink
    'Potato leaf late blight': ['rgb(0, 0, 139)', (0, 0, 139)],  # Dark blue
    'Potato leaf': ['rgb(218, 112, 214)', (218, 112, 214)],  # Orchid
    'Raspberry leaf': ['rgb(255, 0, 255)', (255, 0, 255)],  # Magenta
    'Soyabean leaf': ['rgb(255, 69, 0)', (255, 69, 0)],  # Red-orange
    'Squash Powdery mildew leaf': ['rgb(0, 255, 255)', (0, 255, 255)],  # Cyan
    'Strawberry leaf': ['rgb(255, 255, 0)', (255, 255, 0)],  # Yellow
    'Tomato Early blight leaf': ['rgb(154, 205, 50)', (154, 205, 50)],  # Yellow-green
    'Tomato Septoria leaf spot': ['rgb(0, 0, 255)', (0, 0, 255)],  # Blue
    'Tomato leaf bacterial spot': ['rgb(255, 99, 71)', (255, 99, 71)],  # Tomato
    'Tomato leaf late blight': ['rgb(46, 139, 87)', (46, 139, 87)],  # Sea green
    'Tomato leaf mosaic virus': ['rgb(255, 192, 203)', (255, 192, 203)],  # Pink
    'Tomato leaf yellow virus': ['rgb(173, 255, 47)', (173, 255, 47)],  # Green-yellow
    'Tomato leaf': ['rgb(0, 128, 128)', (0, 128, 128)],  # Teal
    'Tomato mold leaf': ['rgb(128, 0, 0)', (128, 0, 0)],  # Maroon
    'Tomato two spotted spider mites leaf': ['rgb(70, 130, 180)', (70, 130, 180)],  # Steel blue
    'grape leaf black rot': ['rgb(0, 255, 127)', (0, 255, 127)]  # Spring green
}

frequencies = {
    'Apple Scab Leaf': 0,
    'Apple leaf': 0,
    'Apple rust leaf': 0,
    'Bell_pepper leaf spot': 0,
    'Bell_pepper leaf': 0,
    'Blueberry leaf': 0,
    'Cherry leaf': 0,
    'Corn Gray leaf spot': 0,
    'Corn leaf blight': 0,
    'Peach leaf': 0,
    'Potato leaf early blight': 0,
    'Potato leaf late blight': 0,
    'Potato leaf': 0,
    'Raspberry leaf': 0,
    'Soyabean leaf': 0,
    'Squash Powdery mildew leaf': 0,
    'Strawberry leaf': 0,
    'Tomato Early blight leaf': 0,
    'Tomato Septoria leaf spot': 0,
    'Tomato leaf bacterial spot': 0,
    'Tomato leaf late blight': 0,
    'Tomato leaf mosaic virus': 0,
    'Tomato leaf yellow virus': 0,
    'Tomato leaf': 0,
    'Tomato mold leaf': 0,
    'Tomato two spotted spider mites leaf': 0,
    'grape leaf black rot': 0
}

model = YOLO("best.pt","v8")

def draw_bboxes(image, bboxes ):
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[0:4]
        class_name = names[bbox[5]]
        color = colors[class_name][1]
    
        if class_name in frequencies : 
            frequencies[class_name] +=1
        else : 
            frequencies[class_name] =1   
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)

        



st.title('Plants (Apple,tomato,corn,..) disease detector using yolov8 ')
image  = st.file_uploader("**put your image for examintion  :** ")


if image is not None :
    image = Image.open(image) 
    #image = image.filter(ImageFilter.MedianFilter(5))
    image_np = np.array(image)
    
    result = model.predict(source=image_np,conf=0.25,save=False)
    names =result[0].names 
    data = result[0].boxes.data.numpy()
    xyxy = data[:,:]
    button = st.button("start analyzing .." , type="primary")

    if button : 

        draw_bboxes(image, xyxy)
        image.save("output.png")
        



        x = list(frequencies.values())
        y = list(frequencies.keys())
        colors_list = [colors[key][0] for key in y]
        # Create a bar plot
        fig = go.Figure(data=[go.Bar(x=y, y=x, marker_color=colors_list)])

        # Display image in the first column
        
        st.image("output.png", caption='Annotated Image', use_column_width=True)
        st.download_button(
                label="Download image",
                data=BytesIO(image.tobytes()),
                file_name="result_image.jpg",
                key="download_button",
                help="Click to download t image.",
            )

        # Display frequencies in the second column

        st.plotly_chart(fig, use_container_width=True)
