#!/usr/bin/env python3
'''ChessToFEN main file'''
import string
import time
import os
import pyautogui
from cv2 import cv2
import click
from colorama import Fore, Style
from chessClassifier import predict_pieces, convert_to_fen


def capture_screen(screenshot_delay):
    """Captures the screen and saves it as a png file"""
    # Wait for screenshot_delay seconds
    time.sleep(screenshot_delay)
    captured_screen = pyautogui.screenshot()
    # Save image to /temp directory if it exists if not create temp directory
    if os.path.exists("temp"):
        captured_screen.save("temp/captured_screen.png")
    else:
        os.mkdir("temp")
        captured_screen.save("temp/captured_screen.png")


def find_board():
    """Attempts to find a chess board in the image"""
    # Check if temp directory exists and if captured_screen.png exists
    if not os.path.exists("temp/captured_screen.png"):
        print("No screen captured")
        raise FileNotFoundError("No image found")
    # Read the image
    image = cv2.imread("temp/captured_screen.png")
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
    cropped_image = image[y: y + h, x: x + w]
    # If cropped image is too small, no board was found
    if cropped_image.shape[0] < 100 or cropped_image.shape[1] < 100:
        print("No board found")
        raise ValueError("Unable to find board in image please try again")
    # Save cropped image to temp directory
    cv2.imwrite("temp/cropped_board.png", cropped_image)


def crop_out_pieces():
    """From the cropped board image, crop out each square and save it to the data/captures directory"""
    # Check if cropped_board.png exists
    if not os.path.exists("temp/cropped_board.png"):
        raise FileNotFoundError("No image found")
    # Read the image
    image = cv2.imread("temp/cropped_board.png")
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
                int(i * square_length): int((i + 1) * square_length),
                int(j * square_width): int((j + 1) * square_width),
            ]
            file_name = file_names[j] + rank_names[i]
            cv2.imwrite("data/captures/" + file_name + ".png", square)


def display_menu():
    '''Displays the main menu'''
    click.echo()
    click.echo(f"{Fore.YELLOW}Welcome to ChessToFEN!{Style.RESET_ALL}")
    click.echo("Select an option:")
    click.echo(f"{Fore.CYAN}1{Style.RESET_ALL}. Capture screen")
    click.echo(f"{Fore.CYAN}2{Style.RESET_ALL}. Detect board")
    click.echo(f"{Fore.CYAN}3{Style.RESET_ALL}. Crop out pieces")
    click.echo(f"{Fore.CYAN}4{Style.RESET_ALL}. Predict pieces")
    click.echo(f"{Fore.CYAN}5{Style.RESET_ALL}. Convert to FEN")
    click.echo(f"{Fore.CYAN}6{Style.RESET_ALL}. Settings")
    click.echo(f"{Fore.RED}0{Style.RESET_ALL}. Exit")


def display_settings(screenshot_delay):
    '''Displays the settings menu'''
    click.echo(f"{Fore.YELLOW}Settings{Style.RESET_ALL}")
    click.echo("Select an option:")
    click.echo(
        f"{Fore.CYAN}1{Style.RESET_ALL}. Change screenshot delay {Fore.GREEN}(current: {screenshot_delay}){Style.RESET_ALL}")
    click.echo(f"{Fore.RED}0{Style.RESET_ALL}. Back")


def handle_option_1(screenshot_delay):
    '''Handles Option 1'''
    capture_screen(screenshot_delay)
    click.echo(
        f"{Fore.GREEN}Screen captured, saved to temp/captured_screen.png{Style.RESET_ALL}")


def handle_option_2():
    '''Handles Option 2'''
    # Attempt to find board
    try:
        find_board()
    # If no image is found or no board is found in the image, display error
    # message
    except FileNotFoundError:
        click.echo(
            f"{Fore.RED}No image found! Please capture a screen first.{Style.RESET_ALL}")
    except ValueError:
        click.echo(
            f"{Fore.RED}Unable to find board in image! Please try again.{Style.RESET_ALL}")
    # If board is found, display success message and display the cropped board
    else:
        click.echo(
            f"{Fore.GREEN}Board found, saved to temp/cropped_board.png{Style.RESET_ALL}")
        # Display the cropped board
        click.echo(
            f"{Fore.YELLOW}Displaying cropped board...{Style.RESET_ALL}")
        cropped_board = cv2.imread("temp/cropped_board.png")
        cv2.imshow("Cropped Board", cropped_board)
        # Close the window when any key is pressed
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        click.echo(
            f"{Fore.BLUE}If cropped image is NOT correct please try capturing the screen again{Style.RESET_ALL}")


def handle_option_3():
    '''Handles Option 3'''
    # Attempt to crop out pieces
    try:
        crop_out_pieces()
    # If no image is found or no board is found in the image, display error
    # message
    except FileNotFoundError:
        click.echo(
            f"{Fore.RED}No image found! Please capture a screen first.{Style.RESET_ALL}")
    # If pieces are cropped, display success message
    else:
        click.echo(
            f"{Fore.GREEN}Pieces cropped, saved to data/captures{Style.RESET_ALL}")


def handle_option_4():
    '''Handles Option 4'''
    # Attempt to predict pieces
    chess_board = predict_pieces()
    click.echo(f"{Fore.GREEN}Pieces predicted!{Style.RESET_ALL}")
    click.echo(
        f"{Fore.YELLOW}Displaying predicted 2D array...{Style.RESET_ALL}")
    for row in chess_board:
        click.echo(row)
    return chess_board


def handle_option_5(chessboard):
    '''Handles Option 5'''
    # Attempt to convert to FEN
    fen = convert_to_fen(chessboard)
    # Display FEN string
    click.echo(f"{Fore.GREEN}FEN string generated!{Style.RESET_ALL}")
    click.echo(f"{Fore.YELLOW}Displaying FEN string...{Style.RESET_ALL}")
    click.echo(fen)


def handle_option_6(screenshot_delay):
    '''Handles Option 6'''
    # Display settings menu
    while True:
        display_settings(screenshot_delay)
        try:
            choice = int(
                click.prompt(
                    f"{Fore.YELLOW}Enter your choice:{Style.RESET_ALL}",
                    type=int))
            if choice == 0:
                return screenshot_delay
            elif choice == 1:
                user_screenshot_delay = float(
                    click.prompt(
                        f"{Fore.YELLOW}Enter screenshot delay:{Style.RESET_ALL}",
                        type=float))
                # Check if screenshot delay is valid
                if user_screenshot_delay > 0:
                    screenshot_delay = user_screenshot_delay
                    click.echo(
                        f"{Fore.GREEN}Screenshot delay changed to {screenshot_delay}{Style.RESET_ALL}")
                else:
                    click.echo(
                        f"{Fore.RED}Invalid screenshot delay! Please enter a positive number.{Style.RESET_ALL}")
            else:
                click.echo(
                    f"{Fore.RED}Invalid choice! Please select a valid option.{Style.RESET_ALL}")
        except ValueError:
            click.echo(
                f"{Fore.RED}Invalid input! Please enter a number.{Style.RESET_ALL}")


@click.command()
def main():
    '''Main function'''
    # Initialize variables
    chessboard = None
    screenshot_delay = 0.5
    # Display main menu
    while True:
        display_menu()
        try:
            choice = int(
                click.prompt(
                    f"{Fore.YELLOW}Enter your choice:{Style.RESET_ALL}",
                    type=int))
            if choice == 0:
                click.echo(f"{Fore.RED}Goodbye!{Style.RESET_ALL}")
                # Delete temp directory recursively if it exists
                if os.path.exists("temp"):
                    os.system("rm -rf temp")
                break
            elif choice == 1:
                handle_option_1(screenshot_delay)
            elif choice == 2:
                handle_option_2()
            elif choice == 3:
                handle_option_3()
            elif choice == 4:
                chessboard = handle_option_4()
            elif choice == 5:
                handle_option_5(chessboard)
            elif choice == 6:
                screenshot_delay = handle_option_6(screenshot_delay)
            else:
                click.echo(
                    f"{Fore.RED}Invalid choice! Please select a valid option.{Style.RESET_ALL}")
        except ValueError:
            click.echo(
                f"{Fore.RED}Invalid input! Please enter a number.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
