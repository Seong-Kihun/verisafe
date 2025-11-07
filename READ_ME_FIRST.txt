====================================
  VeriSafe - 3 Steps to Start!
====================================

First Time Setup (Do Once):
----------------------------

1. Double-click: STEP1_database.bat
   → Starts PostgreSQL and Redis
   → Wait 30 seconds

2. Double-click: STEP2_install.bat
   → Installs Python packages
   → Wait 5-10 minutes (grab coffee!)

3. Double-click: STEP3_setup.bat
   → Creates database tables
   → Wait 1 minute


Every Time You Start:
---------------------

1. Double-click: STEP1_database.bat
   (Only if Docker is not running)

2. Double-click: START.bat
   → Starts backend server

3. Open browser: http://localhost:8000/docs
   → You should see API documentation


Need Docker?
-----------
Download from: https://www.docker.com/products/docker-desktop/


Problems?
---------
1. Make sure Docker Desktop is running
2. Make sure you ran all 3 STEP files (first time only)
3. Check if port 8000 is available (close other programs)


That's it! Enjoy VeriSafe!
