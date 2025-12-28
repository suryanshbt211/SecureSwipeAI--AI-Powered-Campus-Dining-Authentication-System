# How to View Stored Data

## Quick View Script

I have created a utility script (`view_data.py`) that displays all your stored information in a readable format.

### How to Use

Simply run:
```bash
python3 view_data.py
```

Or on Windows:
```bash
python view_data.py
```

### What It Shows

The script displays:

1. **Users DataFrame** - All registered users with:
   - User ID, University ID, Name, Email
   - Current meal swipe balance
   - Number of registered face embeddings
   - Role and registration date
   - Summary statistics

2. **Transactions DataFrame** - All transaction history:
   - Transaction IDs and types
   - User information
   - Amount changes and balances
   - Timestamps
   - Summary statistics

3. **Face Embeddings** - Face recognition data:
   - Which users have embeddings stored
   - Number of embeddings per user
   - Embedding dimensions and types
   - Total embeddings count

4. **Complete User Information** - Detailed view for each user:
   - All basic information
   - Current meal swipe balance
   - Face recognition registration status
   - Complete transaction history

## Example Output

When you run the script, you'll see output like:

```
================================================================================
  USERS DATAFRAME
================================================================================

📊 Total Users: 4

User #1:
  ├─ User ID:        user-639b549f
  ├─ University ID:  900428564
  ├─ Name:           Taha Sarwat
  ├─ Email:           sarwt01@pfw.edu
  ├─ Meal Swipes:    6
  ├─ Registered Faces: 4
  ├─ Role:           user
  └─ Created:        2025-10-31T23:16:09.283953
...
```

## Direct File Access

You can also view the data directly:

### CSV Files (Human Readable)
- **Users**: `data/users.csv` - Open in Excel, Google Sheets, or any text editor
- **Transactions**: `data/transactions.csv` - Open in Excel, Google Sheets, or any text editor

### Pickle File (Binary)
- **Face Embeddings**: `data/faces.pkl` - This is a binary file, use the script to view it

## Verification Checklist

After running your app, verify:

✅ **Users DataFrame** contains:
- [ ] Name
- [ ] University ID
- [ ] Email
- [ ] Meal swipe balance (default: 8)
- [ ] Registration timestamp

✅ **Face Embeddings** are stored:
- [ ] Each registered user has embeddings
- [ ] Number of embeddings matches registered_faces count
- [ ] Embeddings are 128-dimensional arrays

✅ **Transactions** are recorded:
- [ ] Initial balance transaction when user registers
- [ ] Meal swipe transactions when user accesses system
- [ ] Balance updates correctly

## Quick Test

1. Register a new user through the web app
2. Run `python3 view_data.py` to see the new user
3. Use the meal swipe feature
4. Run the script again to see the transaction recorded
5. Check that the balance decreased by 1

## Troubleshooting

**If the script shows "file not found":**
- Make sure you've run the Flask app at least once
- Check that the `data/` directory exists
- Verify files are created in the correct location

**If embeddings are missing:**
- Make sure you uploaded at least 3 face images during registration
- Check that face detection worked (you should see "registered_faces" count > 0)

**If transactions are missing:**
- Verify that meal swipe transactions are being processed
- Check that the Flask app is saving data correctly

