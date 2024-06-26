"""This create prediction page"""

# Import necessary module
from math import sqrt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_log_error, mean_squared_error

def app(df):
    # Use markdown to give title
    st.markdown("<p style='color:red; font-size: 30px'>This app uses <b>Linear regression</b> to predict the price of a car based on your inputs.</p>", unsafe_allow_html=True)

    # Create a section for user to input data.
    st.header("Select Values:")
    
    # Create sliders.
    ProductName_Cabbage = st.slider("Cabbage", float(df["ProductName_Cabbage - Retail"].min()), float(df["ProductName_Cabbage - Retail"].max()))
    ProductName_cauliflower = st.slider("ProductName_Cauliflower - Retail", float(df["ProductName_Cauliflower - Retail"].min()), float(df["ProductName_Cauliflower - Retail"].max()))
    ProductName_Tomatoes = st.slider("ProductName_Tomatoes - Retail", float(df["ProductName_Tomatoes - Retail"].min()), float(df["ProductName_Tomatoes - Retail"].max()))

    # Creat two radio selection for 0 1 input.
    perKilo = st.radio("Is the food is per kilos?", ("Yes", "No"))
    rawprod = st.radio("Is the food is raw product?", ("Yes", "No"))

    # Modify radio data.
    if (perKilo == "Yes"):
        perKilo = 1;
    else:
        perKilo = 0;
    
    if (rawprod == "Yes"):
        rawprod = 1;
    else:
        rawprod = 0;

    # Create a list of all input.
    feature_list = [[ProductName_Cabbage,ProductName_cauliflower, ProductName_Tomatoes, perKilo, rawprod]]
    
    # Create a button to predict.
    if st.button("Predict"):
        # Get the all values from predict funciton.
        score, pred_price, rsquare_score, mae, msle, rmse = predict(df, feature_list)

        # Display all the values.
        st.success(f"The predicted price of the car: ${int(pred_price):,}")
        st.info(f"Accuracy score of this model is: {score:.2%}")
        st.info(f"R-squared score of this model is: {rsquare_score:.2}")
        st.info(f"Mean absolute error of this model is: {mae:.3f}")
        st.info(f"Mean squared log error of this model is: {msle:.3f}")
        st.info(f"Root mean squared error of this model is: {rmse:.3f}")

@st.cache()
def predict(df, feature_list):
    # Create feature and target variable
    X = df.drop(columns = ['Price'])
    y = df['Price']

    # Split the data in train test.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # perform preprocesscing part
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Create the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Store score and predicted price in a variable.
    score = model.score(X_train, y_train)
    pred_price = model.predict(feature_list)
    pred_price = pred_price[0]

    # Calculate statical data from the model.
    y_test_pred = model.predict(X_test)
    rsquare_score = r2_score(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    msle = mean_squared_log_error(y_test, y_test_pred)
    rmse = sqrt(mean_squared_error(y_test, y_test_pred))

    # Return the values.
    return score, pred_price, rsquare_score, mae, msle, rmse