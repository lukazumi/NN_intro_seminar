"""
Luke Jackson I-Tap 2017
"""

import cv2
import threading
import os
import copy
import sys
import argparse
import time
import shutil

save = False


def ready_save():
    global save
    save = True


def video_feed(args):
    global save
    face_front_cascade = cv2.CascadeClassifier("haar-cascades/haarcascade_frontalface_default.xml")
    face_profile_cascade = cv2.CascadeClassifier("haar-cascades/haarcascade_profileface.xml")
    cascade_adjust = 25

    cnt = 0

    # get class name and append class
    classname = "-1"
    if not os.path.exists(args.output_directory):
        classname = "0"
        os.makedirs(args.output_directory)
    else:
        with open(os.path.join(args.output_directory, "class_lookup.txt")) as text_file:
            c = list(text_file)[-1]
            classname = int(c.split(",")[0])+1
            classname = str(classname)
    with open(os.path.join(args.output_directory, "class_lookup.txt"), "a+") as text_file:
        text_file.write(classname+","+args.label+"\n")

    cap = cv2.VideoCapture(0)

    # ready first capture.
    threading.Timer(args.photo_interval, ready_save).start()
    time.sleep(2)  # web camera warm up
    # run the camera loop
    while True:
        # exit the camera loop prematurely with the q key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):  # pause
            print('p')
            cv2.waitKey(-1)  # wait until press again
        screen_txt = '%s : %s / %s' % (args.label, cnt + 1, args.number_of_samples)
        # Capture frame-by-frame
        ret, frame = cap.read()
        # print status
        no_text_frame = copy.copy(frame)
        no_text_frame = cv2.cvtColor(no_text_frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, (0, 0), (720, 50), (255, 255, 255), -1)
        cv2.putText(frame, screen_txt, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.imshow('frame', frame)
        if save:
            photo_output_directory_raw = '%s\\raw\\%s\\%s%s' % \
                                         (args.output_directory, classname, args.label, cnt)
            photo_output_directory_crop = '%s\\cropped\\%s\\%s%s' % \
                                          (args.output_directory, classname, args.label, cnt)
            if not os.path.exists(os.path.dirname(photo_output_directory_raw)):
                os.makedirs(os.path.dirname(photo_output_directory_raw))
            if not os.path.exists(os.path.dirname(photo_output_directory_crop)):
                os.makedirs(os.path.dirname(photo_output_directory_crop))

            # CROPPED
            # detect object (as per input har-cascade)
            faces = face_front_cascade.detectMultiScale(frame, 1.3, 5)
            profilefaces = face_profile_cascade.detectMultiScale(frame, 1.3, 5)

            #print(type(faces))
            #print(type(profilefaces))

            # failed to detect a face. Save is still true, so next iteration will try again.
            if len(faces) < 1 and len(profilefaces) < 1:
                continue

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                save_photos(x, y, w, h, no_text_frame, cascade_adjust, photo_output_directory_crop,
                            photo_output_directory_raw)
            if len(profilefaces) > 0:
                (x, y, w, h) = profilefaces[0]
                save_photos(x, y, w, h, no_text_frame, cascade_adjust, photo_output_directory_crop,
                            photo_output_directory_raw, "p.jpg")
            save = False
            cnt = cnt + 1
            # when all samples are taken, move to the next label and start from the first sample
            if cnt == args.number_of_samples:
                print("    Fin")
                break
            else:
                # schedule the next photo shoot
                threading.Timer(args.photo_interval, ready_save).start()

    cap.release()
    cv2.destroyAllWindows()
    shutil.copyfile(os.path.join(args.output_directory, "class_lookup.txt"),
                    os.path.join(args.output_directory, "cropped", "class_lookup.txt"))
    shutil.copyfile(os.path.join(args.output_directory, "class_lookup.txt"),
                    os.path.join(args.output_directory, "raw", "class_lookup.txt"))
    return


def save_photos(x, y, w, h, frame, cascade_adjust, photo_output_directory_crop, photo_output_directory_raw,
                apnd=".jpg"):
    photo_output_directory_crop = photo_output_directory_crop + apnd
    photo_output_directory_raw = photo_output_directory_raw + apnd
    # stock
    # roi = no_text_frame[y:y + h, x:x + w]
    # a little further cropping goes a long way
    roi_with_margins = frame[(y - cascade_adjust): y + (h + cascade_adjust),
                       (x - cascade_adjust): x + (w + cascade_adjust)]
    cv2.imwrite(photo_output_directory_crop, roi_with_margins)
    print(photo_output_directory_crop)

    # RAW
    cv2.imwrite(photo_output_directory_raw, frame)
    print(photo_output_directory_raw)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('output_directory', type=str, help='Directory to save captured photos.')
    parser.add_argument('label', type=str, help='The class name, in other words what are you taking photo off.')
    parser.add_argument('--number_of_samples', type=int, help='number of photos per class.', default=500)
    parser.add_argument('--photo_interval', type=float, help='time between photo shoots', default=.1)
    return parser.parse_args(argv)


if __name__ == '__main__':
    video_feed(parse_arguments(sys.argv[1:]))
