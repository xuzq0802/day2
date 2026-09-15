# day2

# Environment Setup

## Step 1 - create virtual environment
1. create virtual environment
> python -m venv .venv
2. install libraries / dependencies
> source .venv/bin/activate

## Step 2 - install dependencies/ Python Libraries
1. create a requirements.txt file
2. add, each on a new line, openai, streamlit, python-dotenv
3. run:
> pip install -r requirements.txt

## Step 3 - create a .env file to store our secrets
1. create .env file
> touch .env
2. add the api key to the .env file
> OPENAI_API_KEY = ""

## create streamlit application
1. create python file 
2. run streamlit server
> streamlit run home.py


# Note - Commit Changes
1. open source control on LHS of screen
2. click the plus sign to stage
3. add a commit message in the message bar
4. click commit
5. sync changes
6. check github repository to confirm