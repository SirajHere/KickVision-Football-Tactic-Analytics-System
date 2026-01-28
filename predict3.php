<?php
include "db.php"; // Assuming you're including your DB connection here

session_start();
if (!isset($_SESSION['user'])) {
    header("Location: login.php");
    exit;
}

// Check if the form is submitted and a file is uploaded
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_FILES['file'])) {
    // Print the details of the file uploaded
    echo '<pre>';
    print_r($_FILES['file']);
    echo '</pre>';
    
    // Optionally, you can use var_dump to get more detailed info
    // var_dump($_FILES['file']);
    
    // If you want to process the uploaded file, you can do that here
    // For example, move the file to a permanent location:
    $targetDirectory = "uploads/";  // You can set a folder where the file will be moved
    $targetFile = $targetDirectory . basename($_FILES['file']['name']);
    
    // Move the uploaded file from temporary location to your target directory
    if (move_uploaded_file($_FILES['file']['tmp_name'], $targetFile)) {
        echo "File has been uploaded successfully.";
    } else {
        echo "Sorry, there was an error uploading your file.";
    }
} else {
    echo "No file uploaded.";
}
?>
