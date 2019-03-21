__author__ = 'Roberta Evangelista'
__email__ = 'roberta.evangelista@posteo.de'

"""Modified from the LinkedIn Learning course: 
Deep Learning: Face Recognition, by Adam Geitgey
 
Uses the face_recognition API by Adam Geitgey"""

import face_recognition
import PIL.Image
import PIL.ImageDraw
from pathlib import Path


def detect_faces_square_box(image_filename):
    """
    Detects faces in an image usign HOG

    :param image_filename: str
            Filename of image
    """

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_filename)

    # Use HOG to locate faces: returns 4 (x,y) points [top, bottom, left, right]
    face_locations = face_recognition.face_locations(image)

    number_of_faces = len(face_locations)
    print("I found {} face(s) in this photograph.".format(number_of_faces))

    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)

    for face_location in face_locations:
        # Print the location of each face in this image.
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom,
                                                                                                    right))

        # Draw a box around the face
        draw = PIL.ImageDraw.Draw(pil_image)
        draw.rectangle([left, top, right, bottom], outline="red")

    # Display the image on screen
    pil_image.show()


def find_landmarks(image_filename):
    """
    Finds landmarks in an image using an already trained model (on human faces)

    :param image_filename: str
            Filename of image
    """

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_filename)

    # Find all facial features in all the faces in the image (dict with name of landmark and (x,y) position)
    face_landmarks_list = face_recognition.face_landmarks(image)

    number_of_faces = len(face_landmarks_list)
    print("I found {} face(s) in this photograph.".format(number_of_faces))

    # Load the image into a Python Image Library object
    pil_image = PIL.Image.fromarray(image)

    # Create a PIL drawing object to be able to draw lines later
    draw = PIL.ImageDraw.Draw(pil_image)

    # Loop over each face
    for face_landmarks in face_landmarks_list:

        # Loop over each facial feature
        for name, list_of_points in face_landmarks.items():
            print("The {} in this face has the following points: {}".format(name, list_of_points))

            # Trace out each facial feature in the image with a line!
            draw.line(list_of_points, fill="red", width=2)

    pil_image.show()


def recognize_known_people_in_unknown_images(known_image_1, known_image_2, known_image_3, unknown_image):
    """
    Use face encoding (affine transformation for alignment and Deep Metric Learning, learning with triplets)
    to extract face location points (interpretation tricky) and compare faces using Euclidean distance between face
    encodings

    :param known_image_1, known_image_2, known_image_3: str
                Filenames of images of known persons (should contain one single person, approxiamtely facing forward)

    :param unknown_image: str
                Filename of unknown image. It can contain more than one person

    """
    # Load known images
    image_of_person_1 = face_recognition.load_image_file(known_image_1)
    image_of_person_2 = face_recognition.load_image_file(known_image_2)
    image_of_person_3 = face_recognition.load_image_file(known_image_3)

    # Get the face encoding of each person. This can fail if no one is found in the photo.
    person_1_face_encoding = face_recognition.face_encodings(image_of_person_1)[0]
    person_2_face_encoding = face_recognition.face_encodings(image_of_person_2)[0]
    person_3_face_encoding = face_recognition.face_encodings(image_of_person_3)[0]

    # Create a list of all known face encodings
    known_face_encodings = [
        person_1_face_encoding,
        person_2_face_encoding,
        person_3_face_encoding
    ]

    # Load the image we want to check
    unknown_image = face_recognition.load_image_file(unknown_image)

    # Get face encodings for any people in the picture - enlarge if too small
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)
    if unknown_face_encodings == []:
        print("Enlarging the picture to detect small faces")
        face_location = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=2)
        unknown_face_encodings = face_recognition.face_encodings(unknown_image, known_face_locations=face_location)

    # There might be more than one person in the photo, so we need to loop over each face we found
    for unknown_face_encoding in unknown_face_encodings:

        # Test if this unknown face encoding matches any of the three people we know
        results = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.6)

        name = "Unknown"
        if results[0]:
            name = "Person 1"
        elif results[1]:
            name = "Person 2"
        elif results[2]:
            name = "Person 3"

        print("Found {} in the photo!".format(name))


def put_makeup(image_filename):
    """Identify face features in an image and put some makeup on

    :param image_filename: str
            Filename of image
    """

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(image_filename)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    # Load the image into a Python Image Library object so that we can draw on top of it and display it
    pil_image = PIL.Image.fromarray(image)

    # Create a PIL drawing object to be able to draw lines later
    d = PIL.ImageDraw.Draw(pil_image, 'RGBA')

    for face_landmarks in face_landmarks_list:
        # The face landmark detection model returns these features:
        #  - chin, left_eyebrow, right_eyebrow, nose_bridge, nose_tip, left_eye, right_eye, top_lip, bottom_lip

        # Draw a line over the eyebrows
        d.line(face_landmarks['left_eyebrow'], fill=(128, 0, 128, 100), width=3)
        d.line(face_landmarks['right_eyebrow'], fill=(128, 0, 128, 100), width=3)

        # Draw over the lips
        d.polygon(face_landmarks['top_lip'], fill=(128, 0, 128, 100))
        d.polygon(face_landmarks['bottom_lip'], fill=(128, 0, 128, 100))

    # Show the final image
    pil_image.show()


def find_similar_faces(test_image_filename, check_similarity_folder):
    """
    Go over .png images in folder to find the most similar face to the test image

    :param test_image_filename: str
            Test image (should contain one single face)
    :param check_similarity_folder:
            Folder where all other images are stored
    """

    # Load the image of the person we want to find similar people for
    known_image = face_recognition.load_image_file(test_image_filename)

    # Encode the known image (one person)
    known_image_encoding = face_recognition.face_encodings(known_image)[0]

    # Variables to keep track of the most similar face match we've found
    best_face_distance = 1.0
    best_face_image = None

    # Loop over all the images we want to check for similar people
    for image_path in Path(check_similarity_folder).glob("*.png"):

        unknown_image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(unknown_image)

        # Get the face distance between the known person and all the faces in this image
        face_distance = face_recognition.face_distance(face_encodings, known_image_encoding)[0]

        # If this face is more similar to our known image than we've seen so far, save it
        if face_distance < best_face_distance:
            best_face_distance = face_distance
            # Extract a copy of the actual face image for display
            best_face_image = unknown_image

    # Display the best match!
    pil_image = PIL.Image.fromarray(best_face_image)
    pil_image.show()


if __name__ == "__main__":

    image_filename = '../Ch03/people.jpg'
    detect_faces_square_box(image_filename)
    find_landmarks(image_filename)
    put_makeup(image_filename)

    recognize_known_people_in_unknown_images(known_image_1='../Ch06/person_1.jpg',
                                             known_image_2='../Ch06/person_2.jpg',
                                             known_image_3='../Ch06/person_3.jpg',
                                             unknown_image='../Ch06/unknown_7.jpg')

    test_image = '../Ch07/test_face.jpg'
    my_folder = '../Ch07/people'
    find_similar_faces(test_image_filename=test_image, check_similarity_folder=my_folder)