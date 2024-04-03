import numpy as np
import gradio as gr


def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)

def calculator(argc):
    num1, operation, num2 = argc[0],argc[1],argc[2]
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        if num2 == 0:
            raise gr.Error("Cannot divide by zero!")
        return num1 / num2
    
with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")
    with gr.Tab("Flip Text"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")

    with gr.Tab("Flip Image"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Flip")

    with gr.Tab('Calculator'):
        gr.Number(label='Age', info='In years, must be greater than 0')

        # num1 = gr.Number(label = "number")
        # oper = gr.Radio(["add", "subtract", "multiply", "divide"]),
        # num2 = gr.Number(label = "number")
        # output  = gr.Textbox(label = "result")
        # cal_btn = gr.Button("Calculator")


    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")

    # cal_btn.click(fn=calculator, inputs=[num1,oper,num2], outputs=output)
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)

demo.launch()
