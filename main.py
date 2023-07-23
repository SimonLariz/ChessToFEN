#!/usr/bin/env python3

import pyautogui
from PIL import Image
import string
import cv2


def capture_screen():
    """Captures the screen and saves it as a png file"""
    screen = pyautogui.screenshot()
    screen.save("screenshot.png")


def find_board(image):
    """Attempts to find a chess board in the image"""
    # Resize the image to 50% of its original size
    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    # Grayscale the image using OpenCV
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply erosion to the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_image = cv2.erode(grayscale_image, kernel)
    # Apply dilation to the image
    dilated_image = cv2.dilate(eroded_image, kernel)
    # Canny edge detection
    canny_image = cv2.Canny(dilated_image, 100, 200)
    # Find contours
    contours, hierarchy = cv2.findContours(
        canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    # Crop image around the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = image[y : y + h, x : x + w]
    # If cropped image is too small, no board was found
    if cropped_image.shape[0] < 100 or cropped_image.shape[1] < 100:
        print("No board found")
        return
    return cropped_image


def crop_out_pieces(image):
    """Given chessboard image, crops out pieces and returns a list of images"""
    # Find length of each square
    square_length = image.shape[0] / 8
    # Find width of each square
    square_width = image.shape[1] / 8

    # Define mapping from index to square
    file_names = string.ascii_lowercase[:8]
    rank_names = string.digits[8:0:-1]
    for i in range(8):
        for j in range(8):
            square = image[
                int(i * square_length) : int((i + 1) * square_length),
                int(j * square_width) : int((j + 1) * square_width),
            ]
            file_name = file_names[j] + rank_names[i]
            cv2.imwrite("data/captures/" + file_name + ".png", square)

    print("Done")

import click
from colorama import Fore, Style

# Function to display the menu options
def display_menu():
    # click.clear()
    click.echo(f"{Fore.YELLOW}Welcome to ChessToFEN!{Style.RESET_ALL}")
    click.echo("Select an option:")
    click.echo(f"{Fore.CYAN}1{Style.RESET_ALL}. Capture screen")
    click.echo(f"{Fore.CYAN}2{Style.RESET_ALL}. Detect board")
    click.echo(f"{Fore.CYAN}3{Style.RESET_ALL}. Crop out pieces")
    click.echo(f"{Fore.CYAN}4{Style.RESET_ALL}. Predict pieces")
    click.echo(f"{Fore.CYAN}5{Style.RESET_ALL}. Convert to FEN")
    click.echo(f"{Fore.CYAN}6{Style.RESET_ALL}. Settings")
    click.echo(f"{Fore.RED}0{Style.RESET_ALL}. Exit")

# Function to handle Option 1
def handle_option_1():
    click.echo(f"{Fore.GREEN}Option 1 selected!{Style.RESET_ALL}")

# Function to handle Option 2
def handle_option_2():
    click.echo(f"{Fore.GREEN}Option 2 selected!{Style.RESET_ALL}")

# Function to handle Option 3
def handle_option_3():
    click.echo(f"{Fore.GREEN}Option 3 selected!{Style.RESET_ALL}")

# Function to handle Option 4
def handle_option_4():
    click.echo(f"{Fore.GREEN}Option 4 selected!{Style.RESET_ALL}")



@click.command()
def main():
    while True:
        display_menu()
        try:
            choice = int(click.prompt(f"{Fore.YELLOW}Enter your choice:{Style.RESET_ALL}", type=int))
            if choice == 0:
                click.echo(f"{Fore.RED}Goodbye!{Style.RESET_ALL}")
                break
            elif choice == 1:
                handle_option_1()
            elif choice == 2:
                handle_option_2()
            elif choice == 3:
                handle_option_3()
            else:
                click.echo(f"{Fore.RED}Invalid choice! Please select a valid option.{Style.RESET_ALL}")
        except ValueError:
            click.echo(f"{Fore.RED}Invalid input! Please enter a number.{Style.RESET_ALL}")

if __name__ == "__main__":
    main()

    '''Main function'''
    '''capture_screen()
    current_image = cv2.imread("screenshot.png")
    board_img = find_board(current_image)
    if board_img is not None:
        crop_out_pieces(board_img)'''
    


if __name__ == "__main__":
    main()
