<?php
include 'db.php';

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $name = $_POST["contactname"];
    $email = $_POST["contactemail"];
    $subject = $_POST["contactsubject"];
    $message = $_POST["contactmessage"];

    $sql = "INSERT INTO contact (contactname, contactemail, contactsubject, contactmessage) 
            VALUES ('$name', '$email', '$subject', '$message')";

    if (mysqli_query($conn, $sql)) {
        echo '<script>alert("Thanks for submitting");</script>';
    } else {
        echo "Error: " . $sql . "<br>" . mysqli_error($conn);
    }
}
?>
<!DOCTYPE html>

<head>
  <meta charset="UTF-8">
  <!-- <title> Responsive Contact Us Form  | CodingLab </title>-->
  <link rel="stylesheet" href="style.css">
  <!-- Fontawesome CDN Link -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.2/css/all.min.css"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="css/contactus.css">
</head>

<body>
  

  <div class="container">
    <div class="content">
      <div class="left-side">
        <div class="address details">
          <i class="fas fa-map-marker-alt"></i>
          <div class="topic">Address</div>
          <div class="text-one">Kalanki</div>
          <div class="text-two">Kathmandu 06</div>
        </div>
        <div class="phone details">
          <i class="fas fa-phone-alt"></i>
          <div class="topic">Phone</div>
          <div class="text-one">+0098 9893 5647</div>
          <div class="text-two">+0096 3434 5678</div>
        </div>
        <div class="email details">
          <i class="fas fa-envelope"></i>
          <div class="topic">Email</div>
          <div class="text-one">kickvisionai@gmail.com</div>
          <div class="text-two">kickvisionai@gmail.com</div>
        </div>
      </div>
      <div class="right-side">
        <!-- Right-side content here -->
        <div class="topic-text">Send us a message</div>
        <p>If you have any work from our website, you can send me a message from here. It's our pleasure to help you.</p>
        <form method="post" action="">
          <div class="input-box">
            <input type="text" name="contactname" placeholder="Enter your name" required>
          </div>
          <div class="input-box">
            <input type="text" name="contactemail" placeholder="Enter your email" required>
          </div>
          <div class="input-box">
            <input type="text" name="contactsubject" placeholder="Enter your Subject" required>
          </div>
          <div class="input-box message-box">
            <textarea name="contactmessage" placeholder="Type your message" required></textarea>
          </div>
          <div class="button">
            <input type="submit" value="Send Now">
          </div>
        </form>
      </div>
    </div>
  </div>
</body>

</html>


