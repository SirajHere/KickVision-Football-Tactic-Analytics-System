<?php
include 'db.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST["name"];
    $address = $_POST["address"];
    $email = $_POST["email"];
    $phone = $_POST["phone"];
    $password = $_POST["password"];
    $confirm_password = $_POST["confirm_password"];

    if ($password !== $confirm_password) {
        echo "Error: Passwords do not match.";
        exit;
    }

    $hashed_password = password_hash($password, PASSWORD_DEFAULT);

    $sql = "INSERT INTO signup (user_name, user_address, user_email, user_phone, user_password) 
            VALUES ('$name', '$address', '$email', '$phone', '$hashed_password')";

    if (mysqli_query($conn, $sql)) {
        echo '<script>alert("Signup Success!");</script>';
        header("Location: login.php");
    } else {
        echo "Error: " . $sql . "<br>" . mysqli_error($conn);
    }
}

?>
<!DOCTYPE html>

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signup Page</title>
    <link rel="stylesheet" href="css/singnu.css">
</head>
<body>
<div class="wrapper">
    <div class="container">
        <div class="col-left">
            <div class="login-text">
                <!-- <h2>Logo</h2>
                <p>
                  Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed eget eros dapibus, ultricies tellus vitae, consectetur tortor. Etiam rutrum placerat
                </p> -->
                <a class="btn" href="index.php">Goto Landing Page</a>
            </div>
        </div>
        
        <form method="post" action="signup.php" >
            <div class="col-right">
                <div class="login-form">
                    <h2>Signup</h2>
                        <p>
                            <input type="text" placeholder="Name" name="name" required />
                        </p>
                        <p>
                            <input type="text" placeholder="Address" name="address" required />
                        </p>
                        <p>
                            <input type="email" placeholder="Email" name="email" required />
                        </p>
                        <p>
                            <input type="text" placeholder="Phone Number" name="phone" required />
                        </p>
                        <p>
                            <input type="password" placeholder="Password" name="password" required />
                        </p>
                        <p>
                            <input type="password" placeholder=" Confirm Password" name="confirm_password" required />
                        </p>
                        <p>
                            <input class="btn" type="submit" value="Sign Up" />
                        </p>
    
                        <p>
                            <a href="login.php">Already Have an Account?</a>
                        </p>
    
                </div>
            </div>
            </form>
        </div>

</div>
</body>
</html>