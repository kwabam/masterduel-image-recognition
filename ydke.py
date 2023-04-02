import numpy as np
import base64
import json


class TypedDeck:
    def __init__(self, main=None, extra=None, side=None):
        if main is None:
            main = []
        if side is None:
            side = []
        if extra is None:
            extra = []
        self.main = main
        self.extra = extra
        self.side = side


def get_ydke(typed_deck):
    return f'ydke://' \
           f'{encode_ydke(typed_deck.main)}!' \
           f'{encode_ydke(typed_deck.extra)}!' \
           f'{encode_ydke(typed_deck.side)}!'


def encode_ydke(passcodes):
    uint8_array = np.array(passcodes, dtype='uint32').view(dtype='uint8')
    return base64.b64encode(uint8_array.tobytes()).decode('ascii')


if __name__ == '__main__':
    passcodes = [44256816, 44256816, 44256816]
    ydke = get_ydke(TypedDeck(main=passcodes))

    print(ydke)
