import logging
import os
import random
import traceback

from src import all_letters
from src.Utils import sample

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def evaluating(decoder, max_length):
    """
    Evaluates the model by generating a specified number of predictions based on random starting sequences.

    Args:
        decoder: The model used for generating predictions.
        max_length (int): The maximum length of the generated sequence.

    Logs:
        Info-level logs include the evaluation progress and any errors encountered during the process.
    """
    logging.info("\n------------\n|   EVAL   |\n------------")

    try:
        while True:
            num_predictions = int(input("Enter the number of predictions: "))

            if num_predictions <= 0:
                logging.warning(
                    "Invalid input: number of predictions must be greater than 0.")
                continue
            if not os.path.exists("generated"):
                os.makedirs(f"generated")
            gen = open(f"generated/Output_{num_predictions}.txt", "a+")

            predictions: list[str] = []

            for i in range(num_predictions):
                # Ensure all_letters is defined
                starting_letters = random.choice(all_letters)
                predicted = sample(decoder, max_length, starting_letters)
                predictions.append(predicted)
                gen.write(predicted + "\n")
                logging.info(f"Prediction {i + 1}: {predicted}")

            gen.close()

            logging.info("------------\n")

    except KeyboardInterrupt:
        logging.info("Evaluation process terminated by user.")
        logging.info("------------")
    except Exception as e:
        logging.error(f"An error occurred: {e} ")
        logging.error(f"An error occurred: {traceback.format_exc()} ")
        logging.info("------------")
