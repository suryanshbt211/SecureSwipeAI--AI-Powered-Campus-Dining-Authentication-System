#!/usr/bin/env python3
"""
Utility script to view all stored data in the SecureSwipe AI system.
This script displays users, transactions, and face embeddings information.
"""

import pandas as pd
import pickle
import os
import numpy as np
from datetime import datetime

# File paths
USERS_FILE = 'data/users.csv'
TRANSACTIONS_FILE = 'data/transactions.csv'
FACES_FILE = 'data/faces.pkl'

def print_separator(char='=', length=80):
    """Print a separator line"""
    print(char * length)

def print_header(title):
    """Print a formatted header"""
    print_separator()
    print(f"  {title}")
    print_separator()

def view_users():
    """Display all users from the DataFrame"""
    print_header("USERS DATAFRAME")
    
    if not os.path.exists(USERS_FILE):
        print("❌ Users file not found!")
        return
    
    try:
        users_df = pd.read_csv(USERS_FILE)
        
        if users_df.empty:
            print("⚠️  No users registered yet.")
            return
        
        print(f"\n📊 Total Users: {len(users_df)}\n")
        
        # Display in a nice table format
        for idx, row in users_df.iterrows():
            print(f"User #{idx + 1}:")
            print(f"  ├─ User ID:        {row['user_id']}")
            print(f"  ├─ University ID:  {row['university_id']}")
            print(f"  ├─ Name:           {row['name']}")
            print(f"  ├─ Email:           {row['email']}")
            print(f"  ├─ Meal Swipes:    {row['balance']:.0f}")
            print(f"  ├─ Registered Faces: {row['registered_faces']}")
            print(f"  ├─ Role:           {row['role']}")
            print(f"  └─ Created:        {row['created_at']}")
            print()
        
        # Summary statistics
        print("\n📈 Summary Statistics:")
        print(f"  ├─ Total Users: {len(users_df)}")
        print(f"  ├─ Regular Users: {len(users_df[users_df['role'] == 'user'])}")
        print(f"  ├─ Admin Users: {len(users_df[users_df['role'] == 'admin'])}")
        print(f"  ├─ Total Meal Swipes: {users_df['balance'].sum():.0f}")
        print(f"  ├─ Average Meal Swipes: {users_df['balance'].mean():.1f}")
        print(f"  └─ Users with 0 swipes: {len(users_df[users_df['balance'] == 0])}")
        
    except Exception as e:
        print(f"❌ Error reading users file: {e}")

def view_transactions():
    """Display all transactions from the DataFrame"""
    print_header("TRANSACTIONS DATAFRAME")
    
    if not os.path.exists(TRANSACTIONS_FILE):
        print("❌ Transactions file not found!")
        return
    
    try:
        transactions_df = pd.read_csv(TRANSACTIONS_FILE)
        
        if transactions_df.empty:
            print("⚠️  No transactions recorded yet.")
            return
        
        print(f"\n📊 Total Transactions: {len(transactions_df)}\n")
        
        # Display transactions
        for idx, row in transactions_df.iterrows():
            print(f"Transaction #{idx + 1}:")
            print(f"  ├─ Transaction ID: {row['transaction_id']}")
            print(f"  ├─ User ID:        {row['user_id']}")
            print(f"  ├─ University ID:  {row['university_id']}")
            print(f"  ├─ Type:           {row['transaction_type']}")
            print(f"  ├─ Amount:         {row['amount']:+d}")
            print(f"  ├─ Balance After:  {row['balance_after']:.0f}")
            print(f"  ├─ Status:         {row['status']}")
            print(f"  └─ Timestamp:      {row['timestamp']}")
            print()
        
        # Summary statistics
        print("\n📈 Transaction Summary:")
        print(f"  ├─ Total Transactions: {len(transactions_df)}")
        print(f"  ├─ Meal Swipes Used: {len(transactions_df[transactions_df['transaction_type'] == 'meal_swipe'])}")
        print(f"  ├─ Initial Balances: {len(transactions_df[transactions_df['transaction_type'] == 'initial_balance'])}")
        print(f"  ├─ Admin Added: {len(transactions_df[transactions_df['transaction_type'] == 'admin_added'])}")
        print(f"  └─ Total Amount: {transactions_df['amount'].sum():+d}")
        
    except Exception as e:
        print(f"❌ Error reading transactions file: {e}")

def view_face_embeddings():
    """Display face embeddings information"""
    print_header("FACE EMBEDDINGS STORAGE")
    
    if not os.path.exists(FACES_FILE):
        print("❌ Face embeddings file not found!")
        return
    
    try:
        with open(FACES_FILE, 'rb') as f:
            faces_data = pickle.load(f)
        
        if not faces_data:
            print("⚠️  No face embeddings stored yet.")
            return
        
        print(f"\n📊 Total Users with Face Embeddings: {len(faces_data)}\n")
        
        # Display embeddings info for each user
        for user_id, embeddings_list in faces_data.items():
            print(f"User ID: {user_id}")
            print(f"  ├─ Number of Embeddings: {len(embeddings_list)}")
            
            if len(embeddings_list) > 0:
                # Get embedding shape
                first_embedding = embeddings_list[0]
                if isinstance(first_embedding, np.ndarray):
                    print(f"  ├─ Embedding Shape: {first_embedding.shape}")
                    print(f"  ├─ Embedding Type: {type(first_embedding).__name__}")
                    print(f"  ├─ Data Type: {first_embedding.dtype}")
                    print(f"  └─ Sample Values (first 5): {first_embedding[:5]}")
                else:
                    print(f"  └─ Embedding Type: {type(first_embedding).__name__}")
            print()
        
        # Summary
        total_embeddings = sum(len(embeddings) for embeddings in faces_data.values())
        print(f"\n📈 Embeddings Summary:")
        print(f"  ├─ Users with embeddings: {len(faces_data)}")
        print(f"  └─ Total embeddings stored: {total_embeddings}")
        
    except Exception as e:
        print(f"❌ Error reading face embeddings file: {e}")

def view_user_transaction_history(user_id=None, university_id=None):
    """Display transaction history for a specific user"""
    if not user_id and not university_id:
        return
    
    print_header("USER TRANSACTION HISTORY")
    
    try:
        transactions_df = pd.read_csv(TRANSACTIONS_FILE)
        users_df = pd.read_csv(USERS_FILE)
        
        # Filter by user
        if university_id:
            user = users_df[users_df['university_id'] == university_id]
            if not user.empty:
                user_id = user.iloc[0]['user_id']
                user_name = user.iloc[0]['name']
            else:
                print(f"❌ User with University ID '{university_id}' not found!")
                return
        else:
            user = users_df[users_df['user_id'] == user_id]
            if not user.empty:
                user_name = user.iloc[0]['name']
            else:
                print(f"❌ User with ID '{user_id}' not found!")
                return
        
        user_transactions = transactions_df[transactions_df['user_id'] == user_id]
        
        if user_transactions.empty:
            print(f"⚠️  No transactions found for {user_name} ({user_id})")
            return
        
        print(f"\n👤 User: {user_name} ({user_id})")
        print(f"📊 Total Transactions: {len(user_transactions)}\n")
        
        for idx, row in user_transactions.iterrows():
            print(f"  {row['timestamp']} | {row['transaction_type']:20s} | "
                  f"Amount: {row['amount']:+3d} | Balance: {row['balance_after']:.0f}")
        
    except Exception as e:
        print(f"❌ Error reading transaction history: {e}")

def view_complete_user_info():
    """Display complete information for each user"""
    print_header("COMPLETE USER INFORMATION")
    
    try:
        users_df = pd.read_csv(USERS_FILE)
        transactions_df = pd.read_csv(TRANSACTIONS_FILE)
        
        with open(FACES_FILE, 'rb') as f:
            faces_data = pickle.load(f)
        
        if users_df.empty:
            print("⚠️  No users registered yet.")
            return
        
        for idx, user_row in users_df.iterrows():
            user_id = user_row['user_id']
            
            print(f"\n{'='*80}")
            print(f"USER #{idx + 1}: {user_row['name']}")
            print(f"{'='*80}")
            
            # Basic Info
            print("\n📋 Basic Information:")
            print(f"  ├─ User ID:        {user_row['user_id']}")
            print(f"  ├─ University ID:  {user_row['university_id']}")
            print(f"  ├─ Name:           {user_row['name']}")
            print(f"  ├─ Email:           {user_row['email']}")
            print(f"  ├─ Role:           {user_row['role']}")
            print(f"  └─ Created:        {user_row['created_at']}")
            
            # Meal Swipes
            print(f"\n🍽️  Meal Swipes:")
            print(f"  └─ Current Balance: {user_row['balance']:.0f}")
            
            # Face Embeddings
            print(f"\n👤 Face Recognition:")
            if user_id in faces_data:
                embeddings = faces_data[user_id]
                print(f"  ├─ Registered:     Yes")
                print(f"  └─ Embeddings:     {len(embeddings)} face embeddings stored")
            else:
                print(f"  └─ Registered:     No face embeddings found")
            
            # Transaction History
            user_transactions = transactions_df[transactions_df['user_id'] == user_id]
            print(f"\n📜 Transaction History ({len(user_transactions)} transactions):")
            if not user_transactions.empty:
                for txn_idx, txn_row in user_transactions.iterrows():
                    print(f"  ├─ {txn_row['timestamp']} | {txn_row['transaction_type']:20s} | "
                          f"{txn_row['amount']:+3d} | Balance: {txn_row['balance_after']:.0f}")
            else:
                print(f"  └─ No transactions yet")
            
            print()
        
    except Exception as e:
        print(f"❌ Error displaying user information: {e}")

def main():
    """Main function to display all data"""
    print("\n" + "="*80)
    print("  SecureSwipe AI - Data Viewer")
    print("  View all stored information in your system")
    print("="*80 + "\n")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        print("❌ Data directory not found! Run the application first to create data files.")
        return
    
    # Display all information
    view_users()
    print("\n")
    view_transactions()
    print("\n")
    view_face_embeddings()
    print("\n")
    view_complete_user_info()
    
    print_separator()
    print("✅ Data viewing complete!")
    print_separator()

if __name__ == '__main__':
    main()

