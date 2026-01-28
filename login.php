<?php
session_start();
include 'db.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $email = $_POST["email"];
    $password = $_POST["password"];

    $sql = "SELECT * FROM signup WHERE user_email = '$email'";

    $result = mysqli_query($conn, $sql);

    if ($result && mysqli_num_rows($result) > 0) {
        $user = mysqli_fetch_assoc($result);

        if (password_verify($password, $user['user_password'])) {
            $_SESSION['user'] = $user;
            if ($user['user_email'] === 'admin@admin.com') {
                header("Location: dashboard.php");
            } else {
                header("Location: homepage.php");
            }
            exit();
        } else {
            echo '<script>alert("Error: Invalid password.")</script>';
        }
    } else {
        echo '<script>alert("Error: User not found.")</script>';
    }
}
?>

<!DOCTYPE html>

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login Page</title>
    <link rel="stylesheet" href="css/login.css">
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

        <div class="col-right">
            <div class="login-form">
                <h2>Login</h2>
                <form method="post" action="login.php">
                    <p>
                        <input type="email" placeholder="Email" name="email" required>
                    </p>
                    <p>
                        <input type="password" placeholder="Password" name="password" required>
                    </p>
                    <p>
                        <input class="btn" type="submit" value="Sign In" />
                    </p>
                    <p>
                        <a href="login.php">Forget password?</a>
                        <a href="signup.php">Create an account.</a>
                    </p>
                </form>
            </div>
        </div>
    </div>
</div>
</body>
</html>
