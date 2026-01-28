<?php
include "db.php";

session_start();
if (!isset($_SESSION['user'])) {
    header("Location: login.php");
    exit;
}

$currentUser = $_SESSION['user'];

// Handle the form submission to update user profile
if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST['update'])) {
    $id = $_SESSION['user']['id'];
    $name = $_POST["name"];
    $mobile_no = $_POST["mobile_no"];
    $email = $_POST["email"];
    $address = $_POST["address"];

    $sql = "UPDATE signup SET user_name = '$name', user_phone = '$mobile_no', user_email = '$email', user_address = '$address' WHERE id = '$id'";

    // Execute the query
    if (mysqli_query($conn, $sql)) {
        // Update the current user's information in the session
        $_SESSION['user']['name'] = $name;
        $_SESSION['user']['mobile_no'] = $mobile_no;
        $_SESSION['user']['email'] = $email;
        $_SESSION['user']['address'] = $address;

        // Redirect back to the editprofile.php page after successful update
        header('Location: editprofile.php');
        exit();
    } else {
        echo "Error updating profile: " . mysqli_error($conn);
    }
}

// Handle delete action
if (isset($_GET['delete_id'])) {
    $delete_id = $_GET['delete_id'];

    // Prepare the SQL query to delete the user's account
    $sql = "DELETE FROM signup WHERE id = '$delete_id'";

    // Execute the query
    if (mysqli_query($conn, $sql)) {
        // Redirect to the logout page after successful deletion
        header('Location: logout.php');
        exit();
    } else {
        echo "Error deleting account: " . mysqli_error($conn);
    }
}
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Edit Profile</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet">
     <link rel="stylesheet" href="css/editprofile.css"> 
</head>

<body>
    <div class="insidecontent">
        <div class="back">
            <a href="homepage.php"><i class="fa-solid fa-backward"> Back</i></a>
        </div>
        <h3>Edit your Information</h3>
        <form method="post" action="editprofile.php">
            <div class="inputsection">
                <div class="name futsal">
                    <input type="text" placeholder="Name" name="name" required />
                </div>
                <div class="contact futsal">
                    <input type="text" placeholder="Phone Number" name="mobile_no" required />
                </div>
                <div class="email futsal">
                    <input type="text" placeholder="Email" name="email" required />
                </div>
                <div class="address futsal">
                    <input type="text" placeholder="Address" name="address" required />
                </div>
            </div>
            <div class="edit">
                <button type="submit" name="update">Save</button>
            </div>
        </form>
        <div class="delete">
            <!-- Add confirmation dialog for delete -->
            <form method="post" action="editprofile.php?delete_id=<?php echo $currentUser['id']; ?>" onsubmit="return confirm('Are you sure you want to delete your account?');">
                <button type="submit">Delete Account</button>
            </form>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js"></script>
</body>

</html>
