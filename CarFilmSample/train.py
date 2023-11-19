import shutil
import traceback
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


def load_data(src,
              target_size=(256, 256)):
    class_dirs = [x for x in Path(src).iterdir()]
    classes = {class_dir.parts[-1]: num for num, class_dir in enumerate(class_dirs)}
    X = []
    Y = []
    for class_dir in class_dirs:
        class_ = class_dir.parts[-1]
        paths = [x for x in class_dir.iterdir()]
        for path in paths:
            img = cv2.imread(str(path))
            img = cv2.resize(img, target_size)
            X.append(img)
            Y.append(classes[class_])
    return np.stack(X), np.stack(Y)


def create_epoch_history(history):
    data = []
    for x in range(len(history.epoch)):
        datum = {'epoch': x,
                 'loss': round(history.history['loss'][x], 2),
                 'val_loss': round(history.history['val_loss'][x], 2),
                 'accuracy': round(history.history['accuracy'][x], 2),
                 'val_accuracy': round(history.history['val_accuracy'][x], 2),
                 'lr': history.history['lr'][x],
                 }
        data.append(datum)
    df = pd.DataFrame(data)
    return df


def create_training_session_entry(history):
    best_epoch = history.history['val_accuracy'].index(max(history.history['val_accuracy']))
    d = {'loss': round(history.history['loss'][best_epoch], 2),
         'val_loss': round(history.history['val_loss'][best_epoch], 2),
         'accuracy': round(history.history['accuracy'][best_epoch], 2),
         'val_accuracy': round(history.history['val_accuracy'][best_epoch], 2),
         'lr': history.history['lr'][best_epoch]}
    return d


def prepare_model(dropout=0.2,
                  fc_layers=(512,),
                  target_size=(256, 256),
                  num_classes=1,
                  classification_layer='sigmoid'):
    base_model = VGG19(weights='imagenet',
                       include_top=False,
                       input_shape=(*target_size, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # Classification layer
    predictions = Dense(num_classes, activation=classification_layer)(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def train_model(x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                model,
                epochs=10,
                lr_factor=0.2,
                cooldown=0,
                min_lr=0.0,
                log_dir='./training',
                checkpoint_dir='./checkpoints'):
    name = datetime.now().strftime("%y-%m-%d_%H:%M:%S")

    tb = TensorBoard(log_dir=str(Path(log_dir).joinpath(name)),
                     update_freq='batch',
                     profile_batch=0)
    mc = ModelCheckpoint(str(Path(checkpoint_dir).joinpath('{epoch:02d}-{val_loss:.2f}.h5')),
                         save_best_only=True,
                         monitor='val_accuracy')
    rlr = ReduceLROnPlateau(monitor='loss',
                            factor=lr_factor,
                            patience=1,
                            mode='min',
                            min_delta=0.0001,
                            cooldown=cooldown,
                            min_lr=min_lr,
                            verbose=1)
    callbacks = [tb, mc, rlr]

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    save = True
    try:
        history = model.fit(x=x_train,
                            y=y_train,
                            validation_data=(x_val, y_val),
                            epochs=epochs,
                            callbacks=callbacks)

    except KeyboardInterrupt:
        response = input('Save model?\n')
        if response == 'true':
            save = True
        else:
            save = False

    if save:
        loss, accuracy = model.evaluate(x_test,
                                        y_test)
        print(loss, accuracy)

        model_dir = Path(f'./models/{name}')
        if not model_dir.exists():
            Path.mkdir(model_dir)
        model_path = model_dir.joinpath('model.h5')

        episode_hist = create_epoch_history(history)
        episode_hist_path = model_dir.joinpath('episode_history.csv')
        episode_hist.to_csv(str(episode_hist_path))
        d = create_training_session_entry(history)
        d['test_loss'] = loss
        d['test_accuracy'] = accuracy
        history_path = './training.csv'
        if not Path(history_path).exists():
            df = pd.DataFrame([d])
        else:
            df = pd.read_csv(history_path)
            df.append(d,
                      ignore_index=True)
        df.to_csv(history_path)
        model.save(str(model_path))
    else:
        [Path.unlink(x) for x in Path(log_dir).iterdir()]
        Path(log_dir).rmdir()


def train_generator(train_gen,
                    val_gen,
                    model,
                    log_dir,
                    checkpoint_dir,
                    loss='binary_crossentropy',
                    optimizer='adam',
                    epochs=10,
                    lr_factor=0.2,
                    cooldown=0,
                    min_lr=0.0):

    tb = TensorBoard(log_dir=str(log_dir),
                     update_freq='batch',
                     profile_batch=0)
    mc = ModelCheckpoint(str(Path(checkpoint_dir).joinpath('{epoch:02d}-{val_loss:.2f}.h5')),
                         save_best_only=True)
    rlr = ReduceLROnPlateau(monitor='loss',
                            factor=lr_factor,
                            patience=1,
                            mode='min',
                            min_delta=0.0001,
                            cooldown=cooldown,
                            min_lr=min_lr,
                            verbose=1)
    callbacks = [tb, mc, rlr]

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    model.fit_generator(train_gen,
                        validation_data=val_gen,
                        epochs=epochs,
                        callbacks=callbacks)
    return model


def prepare_gen(src,
                target_size=(256, 256),
                rotation_range=45,
                horizontal_flip=True,
                vertical_flip=False,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.2,
                zoom_range=0.2,
                mode='train'):
    if mode == 'train':
        gen = ImageDataGenerator(rotation_range=rotation_range,
                                 horizontal_flip=horizontal_flip,
                                 vertical_flip=vertical_flip,
                                 width_shift_range=width_shift_range,
                                 height_shift_range=height_shift_range,
                                 shear_range=shear_range,
                                 zoom_range=zoom_range)
    else:
        gen = ImageDataGenerator()
    g = gen.flow_from_directory(src,
                                target_size=target_size,
                                class_mode='binary')
    return g


def main(args):
    loss, accuracy = None, None
    save = True

    model = prepare_model(fc_layers=args.fc_layers,
                          target_size=args.target_size)
    name = datetime.now().strftime("%y-%m-%d_%H:%M:%S")
    log_dir = Path('./training').joinpath(name)
    if not log_dir.exists():
        Path.mkdir(log_dir)

    checkpoint_dir = Path('./checkpoints')

    if not Path(checkpoint_dir).exists():
        Path.mkdir(Path(checkpoint_dir))
    else:
        files = [x for x in Path(checkpoint_dir).iterdir()]
        if files:
            [Path.unlink(x) for x in files]

    if not args.augment:
        x_train, y_train = load_data(args.train, target_size=args.target_size)
        x_val, y_val = load_data(args.val, target_size=args.target_size)
        x_test, y_test = load_data(args.test, target_size=args.target_size)

        train_model(x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    model,
                    epochs=args.epochs)
    else:
        train_gen = prepare_gen(args.train,
                                target_size=args.target_size,
                                rotation_range=args.rotation_range,
                                zoom_range=args.zoom_range,
                                shear_range=args.shear_range,
                                width_shift_range=args.width_shift_range,
                                height_shift_range=args.height_shift_range,
                                mode='train')
        val_gen = prepare_gen(args.val,
                              target_size=args.target_size,
                              mode='val')
        test_gen = prepare_gen(args.test,
                               target_size=args.target_size,
                               mode='test')
        try:
            model = train_generator(train_gen,
                                    val_gen,
                                    model,
                                    log_dir,
                                    checkpoint_dir,
                                    epochs=args.epochs,
                                    loss=args.loss,
                                    optimizer=args.optimizer)
            loss, accuracy = model.evaluate_generator(test_gen)
            print(loss, accuracy)
        except KeyboardInterrupt:
            response = input('Save model?(y\n): ')
            if response.lower() == 'y':
                save = True
            else:
                save = False
        except:
            save = False
            traceback.print_exc()

    if save:
        model_dir = Path(f'./models/{name}')
        if not model_dir.exists():
            Path.mkdir(model_dir)
        model_path = model_dir.joinpath('model.h5')

        history = model.history
        episode_hist = create_epoch_history(history)
        episode_hist_path = model_dir.joinpath('episode_history.csv')
        episode_hist.to_csv(str(episode_hist_path))
        session = create_training_session_entry(history)
        params = {'test_loss': loss,
                  'test_accuracy': accuracy,
                  'loss_metric': args.loss,
                  'target_size': ', '.join([str(x) for x in args.target_size]),
                  'optimizer': args.optimizer,
                  'fc_layers': ', '.join([str(x) for x in args.fc_layers]),
                  'epochs': args.epochs,
                  'model_path': str(model_path),
                  'augmented': args.augment,
                  'rotation_range': args.rotation_range if args.augment else 0,
                  'zoom_range': args.zoom_range if args.augment else 0,
                  'shear_range': args.shear_range if args.augment else 0,
                  'width_shift_range': args.width_shift_range if args.augment else 0,
                  'height_shift_range': args.height_shift_range if args.augment else 0
                  }
        data = {**session, **params}

        history_path = './training.csv'
        if not Path(history_path).exists():
            df = pd.DataFrame([data])
        else:
            df = pd.read_csv(history_path)
            df = df.append(data,
                           ignore_index=True)
        df.to_csv(history_path)
        model.save(str(model_path))
    else:
        if Path('.') != Path(log_dir):
            shutil.rmtree(log_dir)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('train')
    ap.add_argument('val')
    ap.add_argument('test')
    ap.add_argument('--loss', '-l', default='binary_crossentropy', type=str)
    ap.add_argument('--optimizer', '-o', default='adam', type=str)
    ap.add_argument('--epochs', '-e', default=10, type=int)
    ap.add_argument('--target_size', '-ts', default=(256, 256), type=int, nargs='+')
    ap.add_argument('--fc_layers', '-fc', default=(512,), type=int, nargs='+')
    ap.add_argument('--augment', '-a', action='store_true')
    ap.add_argument('--zoom_range', '-z', default=0.2, type=float)
    ap.add_argument('--shear_range', '-sh', default=0.2, type=float)
    ap.add_argument('--width_shift_range', '-wsr', default=0.1, type=float)
    ap.add_argument('--height_shift_range', '-hsr', default=0.1, type=float)
    ap.add_argument('--rotation_range', '-rr', default=45, type=int)
    args = ap.parse_args()
    args.target_size = tuple(args.target_size)
    main(args)
