import mysql.connector
from getpass import getpass

print("Testing MySQL connection...")

password = getpass("Enter database password: ")

try:
    connection = mysql.connector.connect(
        host="18.136.157.135",
        port=3306,
        user="dm_team2",
        password=password,
        database="project_sales"
    )

    print("SUCCESS: Database connected!")

    cursor = connection.cursor()

    cursor.execute("SELECT DATABASE();")
    print("Database:", cursor.fetchone()[0])

    cursor.execute("SHOW TABLES;")

    print("Tables:")
    for table in cursor:
        print(table[0])

    cursor.close()
    connection.close()

except mysql.connector.Error as e:
    print("DATABASE CONNECTION FAILED")
    print(e)