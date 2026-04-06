# Delete User Feature - Added to Admin Dashboard

## What's New

The admin dashboard now includes a "Delete User" feature that allows administrators to completely remove users from the system.

## Features Added

### 1. Backend (admin_dashboard/views.py)
- New `delete_user()` function that:
  - Deletes user's face images from filesystem
  - Removes user from database
  - Cascades deletion to all related records (face images and detections)
  - Requires admin authentication

### 2. URL Route (admin_dashboard/urls.py)
- Added route: `/dashboard/delete-user/<user_id>/`

### 3. Frontend (dashboard.html)
- New "Registered Users" section showing:
  - Username
  - Registration date
  - Training status
  - Number of face images
  - Number of detections
  - Delete button for each user

## How to Use

1. Login to admin dashboard (admin/admin)
2. Scroll down to "Registered Users" section
3. Click "Delete User" button next to any user
4. Confirm the deletion (warning shows what will be deleted)
5. User and all related data will be permanently removed

## What Gets Deleted

When you delete a user:
- ✅ User account from database
- ✅ All face images from `media/faces/username/` folder
- ✅ All face image records from database
- ✅ All phone detection records for that user
- ✅ Training status

## Security

- Only accessible when logged in as admin
- Confirmation dialog prevents accidental deletion
- Clear warning about permanent deletion

The feature is now live and ready to use!
