<?php
include 'db.php';
?>

<!DOCTYPE html>
<html lang="en">


<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="shortcut icon" href="images/ucl1.png" type="image/x-icon">
  <title>Landing Page</title>

  <!-- font owesome link -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta2/css/all.min.css"
        integrity="sha512-YWzhKL2whUzgiheMoBFwW8CKV4qpHQAEuvilg9FAn5VJUDwKZZxkJNuGM4XkWuk94WCrrwslk8yWNGmY1EduTA=="
        crossorigin="anonymous" referrerpolicy="no-referrer" />

  <!-- liniking css -->
  <link rel="stylesheet" href="css/afterlogins.css">
  <link rel="stylesheet"  href="css/style.css">
</head>

<body>

<!-- header section starts  -->
<header>
  <!-- logo and name on left side -->

  <a href="#" class="logo"><img src="images/ucl1.png" alt="Kick Vision" srcset="">Kick Vision AI</a>

  <div id="menu-bar" class="fas fa-bars"></div>

  <!-- navigation bar starts -->
  <nav class="navbar">
    <a href="Homepage.php">home</a>
    <a href="#about">about</a>
    <a href="#review">services</a>
    <a href="#foot">contact</a>
    <a href="login.php">Login</a>
  </nav>

  <!-- navigation bar completed -->

</header>

<!-- header section ends -->

<!-- home section starts -->
<section class="home" id="home">
  <div class="content">
    <h3>Come and elevate your game with us. Analyze, book, and play!</h3>
    <p>Experience smarter football analytics with us. Our system tracks performances and streamlines futsal bookings, ensuring seamless gameplay. Take charge of your game effortlessly. Welcome to a world where football meets innovation. Analyze, book, and play!</p>
    <a href="#about" class="btn">know more</a>
  </div>
  <div class="image">
    <img src="images/one.jpg" alt="Kick Vision AI">
  </div>
</section>
<!-- home section comes to end -->

<!-- about section start -->
<section class="about" id="about">
  <h1 class="heading">About <span>us</span></h1>
  <div class="whole">
    <div class="img-container">
    <img src="images/k.jpg" alt="Health Mate">
    </div>
    <div class="description">
      <h3>What is  Kick Vision AI?</h3>
      <P>Kick Vision AI is your ultimate football analytics assistant, dedicated to transforming the game. Using advanced AI, it tracks players, goalkeepers, referees, and the ball in real time. It identifies the player in possession, calculates camera movement, and analyzes ball control for both teams. Plus, with our seamless futsal booking feature, scheduling games has never been easier. Experience football like never before—analyze, book, and play!</P>
      <P>fingertips. also for this 
ChatGPT said:
Experience the future of football analytics with Kick Vision AI – where intelligent tracking meets seamless futsal booking. Take control of your game like never before and elevate your football experience. Join Kick Vision AI today and unlock a new era of smart football insights at your fingertips.</P>
    </div>
  </div>
</section>
<!-- about section ends -->
<section class="review" id="review">

  <h1 class="heading"> What we <span>provide?</span> </h1>

  <div class="box-container">

    <div class="box">
      <img src="images/hospital.jpg" alt="">
      <h3></h3>

      <p>Kick Vision AI simplifies futsal booking by seamlessly connecting you with available slots at nearby venues. With just a few taps, you can browse futsal arenas, check real-time availability, and secure your game instantly. Skip the hassle and uncertainty—with Kick Vision AI, booking your next match is as easy as scoring a goal. Take control of your game today!
      </p>
    </div>
    <div class="box">
      <img src="images/pred.png" alt="">
      <h3></h3>

      <p> Kick Vision AI revolutionizes football analytics by tracking every key aspect of the game. By analyzing players, goalkeepers, referees, and ball movement, our advanced algorithms provide real-time insights, including ball possession, camera movement, and team control metrics. This intelligent approach enhances decision-making, offering a deeper understanding of the game. With Kick Vision AI, experience football like never before—smarter, faster, and more immersive!</p>
    </div>
    <div class="box">
      <img src="images/contactus.jpg" alt="">
      <h3></h3>

      <p> We understand that you may have questions or feedback about our
        platform and services, which is why we have a dedicated contact us service to help you out.
        Our customer support team is always ready to assist you with any questions or
        concerns you may have, and we encourage you to reach out to us if you need any assistance
        or have any feedback about our services.
      </p>
    </div>

  </div>

</section>

<!-- review section ends -->

<!-- hsopitals -->
<section id="hospitals">
  <h1>Here are the currently popular  futsals</h1>

  <div class="all-hospitals">
    <div class="indiviudal-hospital">
      <img src="images/medi.jpeg" alt="">
      <h2>Dhuku Futsal Hub</h2>
      <p>Located in Baluwatar, Kathmandu, Dhuku Futsal Hub is a premier futsal facility known for its well-maintained courts and modern amenities. The venue features two indoor futsal courts equipped with LED sports lighting, ensuring optimal playing conditions regardless of weather. Operating daily from 6:00 AM to 9:00 PM, it caters to both casual players and organized tournaments. The hub also offers additional facilities such as swimming pools and ample parking space, enhancing the overall experience for visitors. 




</p>
    </div>
    <div class="indiviudal-hospital">
      <img src="images/gande.jpg" alt="">
      <h2>Futsal Arena</h2>
      <p>With multiple locations, including GAA Hall in Thamel and Boudha in Kathmandu, Futsal Arena provides 7-a-side futsal games for enthusiasts. Operating hours are from 6:00 AM to 8:00 PM, making it convenient for players to schedule matches throughout the day. The venue is known for its well-maintained courts and facilities, attracting both local players and teams for regular practice sessions and friendly matches.</p>
    </div>
    <div class="indiviudal-hospital">
      <img src="images/norvic.jpg" alt="">
      <h2>Grassroots Recreational Center</h2>
      <p>Situated in Mandikatar, Kathmandu, Grassroots Recreational Center offers two futsal fields and operates daily from 6:00 AM to 10:00 PM. It's a favored spot among locals for its accessibility and well-maintained facilities. The center provides a welcoming environment for players of all skill levels, hosting both casual games and organized events.</p>
    </div>
  </div>
</section>



<!-- footer -->

<div class="footer" id="foot">
  <div class="inner-footer">

    <!--  for company name and description -->
    <div class="footer-items">
      <h1>Website Name</h1>
      <p>Kick Vision AI</p>
    </div>

    <!--  for quick links  -->
    <div class="footer-items">
      <h3>Quick Links</h3>
      <div class="border1"></div> <!--for the underline -->
      <ul>
        <a href="Homepage.php"><li>Home</li></a>
        <a href="Contactus.php"><li>Contact</li></a>
        <a href="#about"><li>About</li></a>
      </ul>
    </div>

    <!--  for some other links -->
    <div class="footer-items">
      <h3>Services</h3>
      <div class="border1"></div>  <!--for the underline -->
      <ul>
        <a href="login.php"><li>Futsal Booking</li></a>
        <a href="login.php"><li>Futsal Booking</li></a>
        <a href="login.php"><li>Football Analytics</li></a>
        <a href="login.php"><li>Football Analytics</li></a>
      </ul>
    </div>

    <!--  for contact us info -->
    <div class="footer-items">
      <h3>Contact us</h3>
      <div class="border1"></div>
      <ul>
        <li><i class="fa fa-map-marker" aria-hidden="true"></i>Dillibazar, KTM</li>
        <li><i class="fa fa-phone" aria-hidden="true"></i>123456789</li>
        <li><i class="fa fa-envelope" aria-hidden="true"></i>kickvisionai@gmail.com</li>
      </ul>

      <!--   for social links -->
      <div class="social-media">
        <a href="https://www.instagram.com/"><i class="fab fa-instagram"></i></a>
        <a href="https://www.facebook.com/"><i class="fab fa-facebook"></i></a>
        <a href="https://www.google.com/"><i class="fab fa-google-plus-square"></i></a>
      </div>
    </div>
  </div>



  <!-- scroll top button  -->
  <a href="#home" class="fas fa-angle-up" id="scroll-top"></a>

  <!-- custom js file link  -->
  <script src="js/script.js"></script>

</body>

</html>