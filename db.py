import psycopg2

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="smart_yield",  # Change to your database name if needed
    user="postgres",
    password="70179877Moh#",  # Replace this with your actual PostgreSQL password
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Check the connection
cur.execute("SELECT version();")
print("Connected to PostgreSQL:", cur.fetchone())

# Close the connection
cur.close()
conn.close()
