<?php
$servername = "localhost";
$username = "root";
$password = "";
$databaseName = "futsals_db";

$conn = mysqli_connect($servername, $username, $password);

if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

$sql = "CREATE DATABASE IF NOT EXISTS $databaseName";
if (!mysqli_query($conn, $sql)) {
    echo "Error creating database: " . mysqli_error($conn) . "<br>";
}

$conn = mysqli_connect($servername, $username, $password, $databaseName);

// Create the signup table
$sql = "CREATE TABLE IF NOT EXISTS signup (
    id INT(6) AUTO_INCREMENT PRIMARY KEY,
    user_name VARCHAR(255) NOT NULL,
    user_address VARCHAR(255) NOT NULL,
    user_email VARCHAR(255) NOT NULL,
    user_phone VARCHAR(15) NOT NULL,
    user_password VARCHAR(255) NOT NULL
)";
if (!mysqli_query($conn, $sql)) {
    echo "Error creating table: " . mysqli_error($conn) . "<br>";
}

$sql = "CREATE TABLE IF NOT EXISTS contact(
    id INT(6) AUTO_INCREMENT PRIMARY KEY,
    contactname VARCHAR(255) NOT NULL,
    contactemail VARCHAR(255) NOT NULL,
    contactsubject VARCHAR(255) NOT NULL,
    contactmessage TEXT NOT NULL
)";
if (!mysqli_query($conn, $sql)) {
    echo "Error creating table: " . mysqli_error($conn) . "<br>";
}


$sql = "CREATE TABLE IF NOT EXISTS futsal (
    id INT AUTO_INCREMENT PRIMARY KEY,
    hname VARCHAR(100) NOT NULL,
    regprice DECIMAL(10, 2) NOT NULL,
    hcontact VARCHAR(20) NOT NULL,
    hlocation VARCHAR(200) NOT NULL,
    description TEXT NOT NULL,
    image3 VARCHAR(255) NOT NULL,
    image1 VARCHAR(255) NOT NULL,
    image2 VARCHAR(255) NOT NULL
)";
if (!mysqli_query($conn, $sql)) {
   echo "Error creating table: " . mysqli_error($conn) . "<br>";
}

$sql = "CREATE TABLE IF NOT EXISTS bookedfutsal (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    futsal_id INT NOT NULL,
    name VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    starting_time VARCHAR(20) NOT NULL,
    FOREIGN KEY (user_id) REFERENCES signup(id),
    FOREIGN KEY (futsal_id) REFERENCES futsal(id)
)";

if (!mysqli_query($conn, $sql)) {
    echo "Error creating table: " . mysqli_error($conn) . "<br>";
 }


