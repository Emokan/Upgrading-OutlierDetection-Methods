# models.py
import numpy as np
from sklearn.svm import OneClassSVM, SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import GridSearchCV
from keras.api.models import Model
from keras.api.layers import (
    Dense, Dropout, Input, Conv2D, MaxPool2D, Flatten, 
    Concatenate, RandomRotation, RandomZoom, RandomTranslation, 
    BatchNormalization
)
from keras.api.optimizers import Adam
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau


def create_ocsvm_model(param_grid=None, cv=3):
    if param_grid is None:
        param_grid = {
            'nu': [0.01, 0.05, 0.1, 0.7], 
            'kernel': ['poly', 'rbf', 'sigmoid', 'linear']
        }
    
    # OCSVM - Use GridSearchCV to find the best parameters for the model
    def ocswm_score_func(estimator, X):
        return np.mean(estimator.predict(X) == 1)

    ocsvm_grid = GridSearchCV(OneClassSVM(), param_grid, cv=cv, scoring=ocswm_score_func)
    
    return ocsvm_grid


def create_lof_model(n_neighbors=17, contamination=0.05):
    return LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)


def create_isolation_forest(n_estimators=100, contamination=0.05, random_state=42):
    return IsolationForest(
        n_estimators=n_estimators, 
        contamination=contamination, 
        random_state=random_state, 
        verbose=0
    )


def create_elliptic_envelope(contamination=0.1, random_state=2):
    return EllipticEnvelope(contamination=contamination, random_state=random_state)

def create_hybrid_model(input_shape=(8, 8, 1), feature_shape=None):
    img_input = Input(shape=input_shape)

    # Data Augmentation Layers
    x = RandomRotation(0.05)(img_input)
    x = RandomZoom(0.1)(x)
    x = RandomTranslation(0.1, 0.1)(x)
    x = BatchNormalization()(x)

    # CNN section
    x = Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    cnn_features = Dense(128, activation='relu')(x)
    cnn_features = Dropout(0.3)(cnn_features)

    # Feature input
    feature_input = Input(shape=(feature_shape,))
    feature_dense1 = Dense(64, activation='relu')(feature_input)  
    feature_dense1 = BatchNormalization()(feature_dense1)
    feature_dense = Dense(32, activation='relu')(feature_dense1)
    
    # Combine the two feature groups
    combined = Concatenate()([cnn_features, feature_dense])
    combined = Dense(64, activation='relu')(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)
    output = Dense(11, activation='softmax')(combined)

    # Create the model
    hybrid_model = Model(inputs=[img_input, feature_input], outputs=output)

    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    hybrid_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    
    return hybrid_model


def train_hybrid_model(model, X_train_img, X_train_features, y_train, 
                      X_val_img, X_val_features, y_val, 
                      epochs=20, batch_size=32, class_weights=None):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(
        [X_train_img, X_train_features], y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        validation_data=([X_val_img, X_val_features], y_val),
        class_weight=class_weights,
        verbose=1
    )
    
    return history
