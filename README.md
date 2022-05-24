# Mining Event Logs RPA - MELRPA
Mining Event Logs for Robotic Process Automation: Looking for the why?


## Requirements:
Microsoft Visual C++ 14.0
Graphviz (it can be downloaded from https://www.graphviz.org/download/)


## Before run
You need to have [Python](https://www.python.org/downloads/) installed.

If desired, you can create an isolated installation of the project requirements by creating a [virtual environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.).

## Project initialization

In the project directory, open a terminal and run:

**`python manage.py makemigrations`**

To create a DB model.

**`python manage.py migrate`**

To insert initial data in DB.

**`python manage.py runserver`**

Runs the app in the debug mode. If you want to init in deploy mode, change in the *agosuirpa/settings.py* file, the *DEBUG* mode attribute to False.

## Screenshots storage

To process correctly screenshots associated to the log, they must follow the "image0001" scheme to be sorted alphabetically, since the component classifier will generate a row for each of the images extracted, processing them in alphabetical order and then the information associated to that image will be added as additional columns to the row corresponding to its order in the log. If an image is missing, for example "image0005", there will be an incoherence in the information stored from row 5 onwards.

## Custom training of GUI components classifier

In the case where the option to train the model is used:
Originally the labels of the GUI component images intended for training the CNN for classification are contained in the name of the images themselves as follows:

Example: ._42-android.widget.TextView.png

The label being the last word between the penultimate dot and the last dot. This is how the system will interpret it for CNN training.

The column indicating the cases must be called "Case", the column indicating the activity "Activity" and the column indicating the variant "Variant". In order to carry out a correct preprocessing of the logs before training the decision model.

## Learn More

You can learn more about the deploy of the application in the [Django documentation](https://docs.djangoproject.com/en/4.0/).
