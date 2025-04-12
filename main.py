import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Preprocessing Function
def preprocess(df_path):
    df = pd.read_csv(df_path)
    
    # dteday to datetime
    df['dteday'] = pd.to_datetime(df['dteday'], format='%d-%m-%Y')
    
    # 1. Filling nan values
    # year
    df.loc[df['yr'].isna(), 'yr'] = df.loc[df['yr'].isna(), 'dteday'].apply(lambda x: 1 if x.year == 2019 else 0)
    # month 
    df.loc[df['mnth'].isna(), 'mnth'] = df.loc[df['mnth'].isna(), 'dteday'].dt.month
    # weekday
    df.loc[df['weekday'].isna(), 'weekday'] = df.loc[df['weekday'].isna(), 'dteday'].dt.weekday
    
    # 2. Holiday & Workingday
    def fill_holiday_workingday(row):
        if pd.isna(row['holiday']) and pd.isna(row['workingday']):
            weekday = row['weekday']
            if weekday in [5, 6]:  # 주말
                return 0, 0
            else:  # 평일
                return 0, 1
        elif pd.isna(row['holiday']):
            holiday = 0 if row['workingday'] == 1 else 1
            return holiday, row['workingday']
        elif pd.isna(row['workingday']):
            if row['holiday'] == 0 and row['weekday'] not in [5, 6]:
                return row['holiday'], 1
            else:
                return row['holiday'], 0
        return row['holiday'], row['workingday']
    
    missing_mask = df['holiday'].isna() | df['workingday'].isna()
    rows_to_fill = df[missing_mask]
    filled_values = []
    
    for idx, row in rows_to_fill.iterrows():
        h, w = fill_holiday_workingday(row)
        filled_values.append((h, w))
    
    df.loc[missing_mask, 'holiday'] = [h for h, w in filled_values]
    df.loc[missing_mask, 'workingday'] = [w for h, w in filled_values]
    
    # Season
    # 3. season 결측값 처리
    def get_season(row):
        """
        날짜(월, 일)를 기반으로 계절을 결정하는 함수
        """
        month = row['dteday'].month
        day = row['dteday'].day
        
        # 기존 데이터에서 해당 월/일의 계절 매핑 확인
        same_date_season = df[
            (df['mnth'] == month) & 
            (df['dteday'].dt.day == day) & 
            (df['season'].notna())
        ]['season']
        
        # 같은 월/일의 season 데이터가 있으면 사용
        if not same_date_season.empty:
            return same_date_season.mode().iloc[0]
        
        # 없으면 해당 월의 일자별 계절 분포 확인
        month_seasons = df[
            (df['mnth'] == month) & 
            (df['season'].notna())
        ].groupby(df['dteday'].dt.day)['season'].agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        
        # 가장 가까운 일자의 계절 정보 찾기
        if not month_seasons.empty:
            # 현재 일자와의 차이가 가장 작은 기존 데이터의 일자 찾기
            closest_day = min(month_seasons.index, key=lambda x: abs(x - day))
            if month_seasons[closest_day] is not None:
                return month_seasons[closest_day]
            
        # 위의 방법으로도 찾지 못한 경우, 계절의 일반적인 구분 적용
        if month == 12:
            return 1 if day >= 21 else 4  # 동지(12/21) 이후 겨울
        elif month == 3:
            return 2 if day >= 21 else 1  # 춘분(3/21) 이후 봄
        elif month == 6:
            return 3 if day >= 21 else 2  # 하지(6/21) 이후 여름
        elif month == 9:
            return 4 if day >= 21 else 3  # 추분(9/21) 이후 가을
        elif month in [1, 2]:
            return 1  # 겨울
        elif month in [4, 5]:
            return 2  # 봄
        elif month in [7, 8]:
            return 3  # 여름
        elif month in [10, 11]:
            return 4  # 가을
    
    df.loc[df['season'].isna(), 'season'] = df[df['season'].isna()].apply(get_season, axis=1)
    
    # fill 'temp', 'atemp', 'windspeed', 'hum' with catboost regressor based using other input features
    weather_numeric_features = ['temp', 'atemp', 'windspeed', 'hum']
    features_for_prediction = ['mnth', 'season', 'yr', 'holiday', 'weekday', 'workingday']
    
    # Nan value check
    missing_any = df[weather_numeric_features].isna().any().any()
    if missing_any:
        for col in weather_numeric_features:
            print(f"{col} 처리 전 결측치 수: {df[col].isna().sum()}")
        
        # fillna with catboostregressor
        for target_col in weather_numeric_features:
            if df[target_col].isna().any():
                missing_mask = df[target_col].isna()
                
                # use other feature values for filling nan if availalbe
                available_numeric = [col for col in weather_numeric_features 
                                  if col != target_col and not df[col].isna().any()]
                current_features = features_for_prediction + available_numeric
                
                model = CatBoostRegressor(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    loss_function='RMSE',
                    verbose=False,
                    random_seed=42
                )
                
                # Predict nan values
                model.fit(
                    df[~missing_mask][current_features],
                    df[~missing_mask][target_col]
                )
                
                predictions = model.predict(df[missing_mask][current_features])
                df.loc[missing_mask, target_col] = predictions.astype(int)
        
        print("\n=== 결측치 처리 결과 ===")
        for col in weather_numeric_features:
            print(f"{col} 처리 후 결측치 수: {df[col].isna().sum()}")

    # filling weathersit nan values using catboostclassifier using all weather related features
    if df['weathersit'].isna().any():
        print("\n=== weathersit 결측치 처리 ===")
        print(f"처리 전 결측치 수: {df['weathersit'].isna().sum()}")
        
        weather_all_features = ['temp', 'atemp', 'hum', 'windspeed', 'season', 'mnth']
        
        missing_mask = df['weathersit'].isna()
        train_data = df[~missing_mask].copy()
        predict_data = df[missing_mask].copy()
        
        model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            loss_function='MultiClass',
            verbose=False,
            random_seed=42
        )
        
        model.fit(
            train_data[weather_all_features],
            train_data['weathersit']
        )
        
        predictions = model.predict(predict_data[weather_all_features])
        df.loc[missing_mask, 'weathersit'] = predictions.astype(int)
        print(f"처리 후 결측치 수: {df['weathersit'].isna().sum()}")
    
    # Check dytpe of each column after filling nan values
    print("\n각 컬럼의 현재 데이터 타입:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")
    
    instant = df['instant']
    
    # Remove unessessary values based on correalation analysis
    remove_columns = ['instant', 'dteday', 'atemp', 'weekday', 'season']
    df = df.drop(columns=remove_columns)
        
    # Categorical values one-hot encoding 
    cat_col = ['mnth', 'weathersit']
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    cat_data = df[cat_col]
    encoded_array = encoder.fit_transform(cat_data)

    feature_names = []
    for i, col in enumerate(cat_col):
        categories = encoder.categories_[i]
        for category in categories:
            feature_names.append(f"{col}_{category}")

    df_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
    df = df.drop(columns=cat_col, errors='ignore')
    df = pd.concat([df, df_encoded], axis=1)
    
    return df, instant

# Seed
def same_seed(seed):
    """Set random seed for reproducibility"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Split
def train_valid_split(data, valid_ratio, seed):
    """Split data into training and validation sets"""
    valid_set_size = int(valid_ratio * len(data))
    train_set_size = len(data) - valid_set_size
    train_set, valid_set = torch.utils.data.random_split(data, [train_set_size, valid_set_size],
                                                         generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

# Prediction
def predict(loader, model, device):
    """Generate predictions"""
    model.eval()
    predictions = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            output = model(x)
            predictions.extend(output.cpu().numpy())
    return predictions

# Dataset
class Bike_Sharing_Dataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        else:
            return self.X[idx], self.y[idx]

# Model
class My_Model(nn.Module):
    """Neural Network Model with BatchNorm, Dropout, and GELU activation"""
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        self.model = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            # Third layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            
            # Fourth layer
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(0.0),
            
            # Output layer with positive output
            nn.Linear(32, 1),

        )
        # Weight initialization
        self.apply(self._init_weights)
    
    # Xavier initialization
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)  
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.1) 
    
    def forward(self, x):
        return self.model(x).squeeze(1)

# Metric & Loss
class RMSLELoss(nn.Module):
    def __init__(self, convertExp=True):
        super(RMSLELoss, self).__init__()
        self.convertExp = convertExp
        self.eps = 1e-7  # Small epsilon value to prevent log(0)

    def forward(self, y_pred, y_true):
        if self.convertExp:
            y_true = torch.exp(y_true)
            y_pred = torch.exp(y_pred)
        
        # Clamp predictions to ensure positive values
        y_pred = torch.clamp(y_pred, min=self.eps)
        y_true = torch.clamp(y_true, min=self.eps)
        
        log1 = torch.log(y_true + 1)
        log2 = torch.log(y_pred + 1)
        loss = torch.mean((log1 - log2) ** 2)
        return torch.sqrt(loss)

# Train model
def train_model(loader, model, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Validation
def validate_model(loader, model, criterion, device):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(loader)

# train_valid_split_train
def train_valid_split_train(train_loader, valid_loader, model, config, device):
    """Complete training process"""
    criterion = RMSLELoss(convertExp=False)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_scheduler(optimizer, config)
    
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    train_losses, valid_losses = [], []
    
    print("Starting training...")
    for epoch in range(config['n_epochs']):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        valid_loss = validate_model(valid_loader, model, criterion, device)
        
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config['save_path'])
            print(f"Epoch [{epoch + 1}/{config['n_epochs']}], "
                  f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                  f"LR: {current_lr:.6f} (Saved)")
        else:
            epochs_no_improve += 1
            print(f"Epoch [{epoch + 1}/{config['n_epochs']}], "
                  f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                  f"LR: {current_lr:.6f}")
            
        if epochs_no_improve >= config['early_stop']:
            print(f"Early stopping at epoch {epoch + 1}")
            break
            
    return train_losses, valid_losses, best_valid_loss

#  No train_valid_split_train
def no_split_train(train_loader, model, config, device):
    """Training process without validation"""
    criterion = RMSLELoss(convertExp=False)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_scheduler(optimizer, config)
    
    train_losses = []
    
    print("Starting training...")
    for epoch in range(config['n_epochs']):
        train_loss = train_model(train_loader, model, criterion, optimizer, device)
        
        if scheduler:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        train_losses.append(train_loss)
        
        print(f"Epoch [{epoch + 1}/{config['n_epochs']}], "
              f"Train Loss: {train_loss:.4f}, "
              f"LR: {current_lr:.6f}")
            
    return train_losses

# Scheduler
def get_scheduler(optimizer, config):
    """Get learning rate scheduler"""
    return CosineAnnealingLR(optimizer, T_max=config['T_max'], eta_min=config['eta_min'])

# Dataloader
def get_dataloader(train_set, valid_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

# Configuration (Modified hyperparameters and add weight decay form better performance)
CONFIG = {
    'seed': 711641,
    'valid_ratio': 0.2,
    'n_epochs': 3500,
    'batch_size': 32,
    'learning_rate': 5e-3,
    'early_stop': 300,
    'save_path': './best_model.pth',
    'scheduler': 'cosine',
    'T_max': 500,
    'eta_min': 5e-4,
    'weight_decay': 1e-3
}

CONFIG_FOR_NOSPLIT = {
    'seed': 711641,
    'n_epochs': 1200,
    'batch_size': 32,
    'learning_rate': 5e-3,  # Reduced learning rate
    'save_path': './no_split_best_model.pth',
    'scheduler': 'cosine',
    'T_max': 500,
    'eta_min': 5e-4,
    'weight_decay': 1e-3
}

# Main
if __name__ == '__main__':
    # First train with splitting for checking the overfitting

    # Set seed for reproducibility
    same_seed(CONFIG['seed'])

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, _ = preprocess('./data/train.csv')
    test_df, instant = preprocess('./data/test.csv')

    # 특성과 타겟 분리
    X = train_df.drop(columns=['cnt'])
    y = train_df['cnt']
    X_test = test_df
    
    # 데이터 분할
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=CONFIG['valid_ratio'], random_state=CONFIG['seed'])
    
    # Dataset 생성
    train_dataset = Bike_Sharing_Dataset(X_train, y_train)
    valid_dataset = Bike_Sharing_Dataset(X_valid, y_valid)
    test_dataset = Bike_Sharing_Dataset(X_test)
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # 모델 초기화
    print(X.shape[1])
    model = My_Model(input_dim=X.shape[1]).to(device)
    
    # 모델 훈련
    train_losses, valid_losses, best_loss = train_valid_split_train(train_loader, valid_loader, model, CONFIG, device)
    
    # 최적 모델 로드
    model.load_state_dict(torch.load(CONFIG['save_path']))
    
    # 예측
    predictions = predict(test_loader, model, device)
    
    # 결과 저장
    pred_df = pd.DataFrame({'ID': instant, 'cnt': predictions})
    pred_df.to_csv('./nn_predictions.csv', index=False)
    
    # Loss 곡선 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()
    
    print("Train-Valid splitting training completed. Results saved to nn_predictions.csv")

    print("===================================Start No Splitting Training for final submission==========================================")

    train_df, _ = preprocess('./data/train.csv')
    test_df, instant = preprocess('./data/test.csv')

    X = train_df.drop(columns=['cnt'])
    y = train_df['cnt']
    X_test = test_df
    
    # Only train and test dataset
    train_dataset = Bike_Sharing_Dataset(X, y)
    test_dataset = Bike_Sharing_Dataset(X_test)
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=CONFIG_FOR_NOSPLIT['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG_FOR_NOSPLIT['batch_size'], shuffle=False)
    
    # Model
    model = My_Model(input_dim=X.shape[1]).to(device)
    
    # Training (no validation)
    train_losses = no_split_train(train_loader, model, CONFIG_FOR_NOSPLIT, device)
    
    # Prediction for submission
    predictions = predict(test_loader, model, device)
    
    # Save result
    pred_df = pd.DataFrame({'ID': instant, 'cnt': predictions})
    pred_df.to_csv('./final_nn_predictions.csv', index=False)
    
    # Loss curve visualization
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('no_split_loss_curves.png')
    plt.close()
    
    print("No splitting training completed. Results saved to final_nn_predictions.csv")
