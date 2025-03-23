#! python3
# -*- coding: utf-8 -*-
'''
@File   : NeuralNetworkPricer.py
@Created: 2025/03/22 12:17
@Author : DONG Yihong
'''

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

from . import Pricer, BlackScholesPricer  # To generate training data

class NeuralNetworkPricer(Pricer):
    # train_model: Sequential

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_history = None
        return

    @staticmethod
    def generate_training_data(num_samples=100000):
        """
        Generate synthetic training data for the neural network.
        
        Args:
            num_samples (int): Number of samples to generate.
            
        Returns:
            tuple: A tuple containing the input features and target prices (call, put).
        """
        np.random.seed(42)  # For reproducibility

        # Randomly generate inputs
        S = np.random.uniform(50, 150, num_samples)  # Spot price
        K = np.random.uniform(50, 150, num_samples)  # Strike price
        r = np.random.uniform(0.01, 0.05, num_samples)  # Risk-free rate
        sigma = np.random.uniform(0.1, 0.5, num_samples)  # Volatility
        T = np.random.uniform(0.1, 2, num_samples)  # Time to maturity
        q = np.random.uniform(0, 0.02, num_samples)  # Dividend yield
        
        # Calculate call prices using Black-Scholes
        call_prices = []
        put_prices = []
        for i in range(num_samples):
            call_price, put_price = BlackScholesPricer.option_price(S[i], K[i], r[i], sigma[i], T[i], q[i])
            call_prices.append(call_price)
            put_prices.append(put_price)
        
        # Combine inputs and outputs
        X = np.column_stack((S, K, r, sigma, T, q))
        y = np.column_stack((call_prices, put_prices))
        
        return X, y

    def build_model(self, input_shape):
        """
        Build a Multi-Layer Perceptron (MLP) model with ReLU activation.
        """
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=input_shape),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(2)  # Output layer for call and put prices (no activation for regression)
        ])

        # Adam optimizer with learning rate scheduler
        optimizer = Adam(learning_rate=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')  # Mean Squared Error loss

    @staticmethod
    def lr_scheduler(epoch, lr):
        """
        Learning rate scheduler to reduce learning rate over time.
        """
        if epoch < 10:
            return lr
        else:
            return lr * np.exp(-0.1)

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Train a neural network model on the provided training data.
        
        Args:
            X_train (np.ndarray): Training input features.
            y_train (np.ndarray): Training target prices (call, put).
            X_val (np.ndarray): Validation input features.
            y_val (np.ndarray): Validation target prices (call, put).
            
        Returns:
            tuple: A tuple containing the trained model and history.
        """
        self.build_model(input_shape=(X_train.shape[1],))
        
        lr_scheduler_callback = LearningRateScheduler(__class__.lr_scheduler)
        
        self.model_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[lr_scheduler_callback],
            verbose=1
        )

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model using RMSE.
        """
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        print(f"RMSE: {rmse:.4f}")
        return rmse

    def option_price(self, S, K, r, sigma, T, q):
        """
        Predict the option price using the trained neural network.
        
        Args:
            S (float): Spot price of the underlying asset.
            K (float): Strike price of the option.
            r (float): Risk-free interest rate.
            sigma (float): Volatility of the underlying asset.
            T (float): Time to maturity (in years).
            q (float): Dividend yield.
            
        Returns:
            tuple: A tuple containing the predicted call price and put price.
        """
        # Load the trained model (or train it if not already trained)
        if not self.model:
            X, y = __class__.generate_training_data()
            X_train, X_val = X[:80000], X[80000:]
            y_train, y_val = y[:80000], y[80000:]
            self.train_model(X_train, y_train, X_val, y_val)
        
        # Predict the option price
        input_data = np.array([[S, K, r, sigma, T, q]])
        call_price, put_price = self.model.predict(input_data)[0]
        
        return call_price, put_price
