from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.models import load_model


def extract_frames(fp,
                   target_size=(256, 256),
                   frameskip=24):
    if not isinstance(target_size, tuple):
        target_size = tuple(target_size)

    frames = []

    cap = cv2.VideoCapture(fp)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in tqdm(range(framecount), desc=f'Extracting frames from {str(fp)}'):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        elif frame_num % frameskip == 0:
            frames.append(cv2.resize(frame, target_size))

    return np.array(frames)


def parse_predictions(predictions,
                      frameskip):
    data = []
    for num, prediction in enumerate(predictions):
        if prediction > 0.5:
            class_ = 0
        else:
            class_ = 1
        datum = {'frame_num': num * frameskip,
                 'car': class_}
        data.append(datum)
    df = pd.DataFrame(data)
    return df


def main(args):
    model = load_model(args.model_path)
    frames = extract_frames(args.src,
                            target_size=args.target_size,
                            frameskip=args.frameskip)
    print('Analyzing frames ...')
    predictions = model.predict(frames, verbose=1)
    df = parse_predictions(predictions, args.frameskip)
    cars = df[df['car'] == 1]
    pct_cars = cars.shape[0]/df.shape[0]
    print(f'\n\nPercent Car: {100 * pct_cars}')

    if args.outpath:
        df.to_csv(args.outpath, index=False)
        print(f'Data saved to {args.outpath}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--outpath', '-o', default=None, type=str)
    ap.add_argument('--target_size', '-ts', default=(256, 256), type=int, nargs='+')
    ap.add_argument('--frameskip', '-fs', default=24, type=int)
    ap.add_argument('--model_path', '-m', default='./model.h5', type=str)
    args = ap.parse_args()
    main(args)
