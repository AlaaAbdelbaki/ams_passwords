import logging
import random
import string
import time

from tqdm import tqdm

from src.Utils import (all_letters, progress, progress_percent, sample,
                       time_since_start)

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def testing(decoder, nb_samples, lineTest, percent, max_length):
    """
    Tests the model by generating a specified number of predictions and measuring the accuracy based on a test set.

    Args:
        decoder: The model used for generating predictions.
        nb_samples (int): The number of samples to generate for testing.
        lineTest (list): The list of valid names to check against.
        percent (float): The percentage of the total names to reach for testing.
        max_length (int): The maximum length for the generated sequences.

    Logs:
        Info-level logs include progress, accuracy, and completion details of the testing process.
    """
    logging.info("\n------------\n|   TEST   |\n------------")

    start = time.time()
    accuracy = 0
    predicted = "a"
    predicted_current = []

    if nb_samples > 0:
        logging.info(f"Testing with {nb_samples} samples...")

        # for i in tqdm(range(1, nb_samples + 1), f"Accuracy :{(accuracy/len(lineTest))*100}%, couverage: {(len(predicted_current)/len(lineTest))*100}%", unit="sample"):
        for i in range(1, nb_samples + 1):
            # Adjusted size for starting letters (can be randomized)
            nc = 1

            # Ensure unique predictions
            while predicted in predicted_current:
                starting_letters = "".join(
                    random.choice(all_letters) for _ in range(nc))
                predicted = sample(decoder, max_length,
                                   starting_letters)
                # .lower()
            predicted_current.append(predicted)

            if predicted in lineTest:
                accuracy += 1

            progress(
                total=nb_samples, acc=accuracy, start=start, epoch=i, l=len(
                    lineTest)
            )

        accuracy = 100 * accuracy / nb_samples
        logging.info(f"Accuracy: {accuracy}%")

    else:
        logging.info("Testing with a percentage-based approach...")
        i = 0
        l = len(lineTest)
        p = int(percent / 100 * l)

        while accuracy < p:
            nc = random.randint(1, int(max_length / 2 - 1))

            while predicted in predicted_current:
                starting_letters = "".join(
                    random.choice(all_letters) for _ in range(nc))
                predicted = sample(decoder, max_length,
                                   starting_letters)
                # .lower()

            predicted_current.append(predicted)

            if predicted in lineTest:
                accuracy += 1

            i += 1
            progress_percent(
                totalNames=l, start=start, names=accuracy, p=percent, samplesGenerated=i
            )

        logging.info(
            f"{percent}% of all names ({len(lineTest)}) reached in {i} iterations ({time_since_start(start)} s)...")
