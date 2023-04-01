import json
from time import sleep
import requests
import shutil
import os
from slugify import slugify

base_url = './cardDatabase/yugioh'

class Card:
    def __init__(self, card_data):
        self.name = card_data['name']
        self.id = str(card_data['card_images'][0]['id'])
        self.image_url = card_data['card_images'][0]['image_url']

    def save_card(self):
        save_directory = f"{base_url}/{slugify(self.name, lowercase=False)}-{self.id}"
        try:
            print(f'Saving card {self.name}')
            if not os.path.exists(f'{save_directory}/{self.id}.png'):
                res = requests.get(self.image_url, stream=True)
                res.raise_for_status()
                sleep(.5)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)
                with open(f'{save_directory}/{self.id}.png', 'wb') as save_file:
                    shutil.copyfileobj(res.raw, save_file)
            else:
                print(f"{self.name} already saved")
        except requests.exceptions.HTTPError as e:
            print(f'HTTP error occurred while saving card: {e}')
        except Exception as e:
            print(f'An error occurred while saving card: {e}')

    def __str__(self):
        return f'Name: {self.name}, ID: {self.id}, URL: {self.image_url}'

if __name__ == '__main__':
    # copy cards.php from: https://db.ygoprodeck.com/api/v7/cardinfo.php
    with open('./cardinfo.php') as f:
        cards_json = json.load(f)['data']

    cards = [Card(card_data) for card_data in cards_json]
    for card in cards:
        card.save_card()

    print(len(cards))
