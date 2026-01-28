<?php
include "db.php";

session_start();
if (!isset($_SESSION['user'])) {
    header("Location: login.php");
    exit;
}

$message = '';

if ($_SERVER["REQUEST_METHOD"] === "POST") {
    $futsal_id = $_POST['futsal_id'];
    $user_id = $_SESSION['user']['id'];
    $name = $_POST['name'];
    $date = $_POST['date'];
    $starting_time = $_POST['starting'];

    $insertQuery = "INSERT INTO bookedfutsal (user_id, futsal_id, name, date, starting_time) VALUES ('$user_id', '$futsal_id', '$name', '$date', '$starting_time')";
    if (mysqli_query($conn, $insertQuery)) {
        $message = "Booking successful!";
    } else {
        $message = "Error: " . mysqli_error($conn);
    }
}

if (isset($_GET['futsal_id'])) {
    $futsal_id = $_GET['futsal_id'];
} else {
    // Handle the case when futsal_id is not provided in the URL
    // For example, redirect to an error page or display an error message
    // header("Location: error.php");
    // exit;
}
?>

<!DOCTYPE html>


<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Book Futsal</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta2/css/all.min.css" integrity="sha512-YWzhKL2whUzgiheMoBFwW8CKV4qpHQAEuvilg9FAn5VJUDwKZZxkJNuGM4XkWuk94WCrrwslk8yWNGmY1EduTA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@8/swiper-bundle.min.css" />
    <link rel="stylesheet" href="css/afterlogins.css" />
    <link rel="stylesheet" href="css/bookfutsal.css" />
</head>

<body>
    <header>
        <a href="Homepage.html" class="logo">
            <img src="images/ucl1.png" alt="futsal" srcset="" />Kick Vision AI
        </a>
        <div id="menu-bar" class="fas fa-bars"></div>
        <div class="searchbar">
            <input type="search" name="search" id="search" />
            <i class="fa-solid fa-magnifying-glass"></i>
        </div>
        <nav class="navbar">
            <a href="homepage.php">home</a>
            <a href="Homepage.php#futsals">Futsals</a>
            <a href="index.php#review">services</a>
            <!-- <a href="#gallery">gallery</a> -->
            <a href="contactus.php">contact</a>
            <a href="bookedfutsal.php">Booked Futsal</a>
            <a href="">
                <span><?php echo $_SESSION['user']['user_name']; ?></span>
            </a>
        </nav>
    </header>

    <div class="container">
        <div class="swiper">
            <div class="swiper-wrapper">
                <!-- Add your swiper content here -->
                <div class="swiper-slide"><img src="images/hos4.jpg" alt=""></div>
                <div class="swiper-slide"><img src="images/hos2.jpeg" alt=""></div>
                <div class="swiper-slide"><img src="images/hos5.jpg" alt=""></div>
            </div>
            <!-- Add your swiper pagination and navigation here -->
        </div>
    </div>

    <section id="contact-us">
        <span class="big-circle"></span>
        <div class="form">
            <div class="contact-info">
                <h3 class="heading">Find us here</h3>
                <iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3532.2193819626273!2d85.34623757428604!3d27.71051197618072!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x39eb191aaaaaaaab%3A0x424c7d0a60df9091!2sPashupatinath%20Temple!5e0!3m2!1sen!2snp!4v1708806227371!5m2!1sen!2snp" width="600" height="450" style="border:0;" allowfullscreen="" loading="lazy" referrerpolicy="no-referrer-when-downgrade"></iframe>
                <div class="info">
                    <div class="information">
                        <i class="fas fa-map-marker-alt"></i>
                        <p>Location</p>
                    </div>
                    <div class="information">
                        <i class="fas fa-phone-alt"></i>
                        <p>Contact</p>
                    </div>
                    <div class="information">
                        <i class="fa-sharp fa-solid fa-rupee-sign"></i>
                        <p>Price</p>
                    </div>
                </div>
                <div class="social-media">
                    <p>Connect with us :</p>
                    <div class="social-icons">
                        <a href="https://www.facebook.com/">
                            <i class="fab fa-facebook-f"></i>
                        </a>
                        <a href="https://www.twitter.com/">
                            <i class="fab fa-twitter"></i>
                        </a>
                        <a href="https://www.instagram.com/">
                            <i class="fab fa-instagram"></i>
                        </a>
                        <a href="https://www.linkedin.com/">
                            <i class="fab fa-linkedin-in"></i>
                        </a>
                    </div>
                </div>
            </div>
            <div class="contact-form">
                <form method="post" action="bookfutsal.php">
                    <span class="circle one"></span>
                    <span class="circle two"></span>
                    <h3 class="heading">Book <span>Here</span></h3>
                    <input type="hidden" name="futsal_id" value="<?php echo htmlspecialchars($futsal_id); ?>">
                    <div class="input-container">
                        <input type="text" name="name" class="input" placeholder="Name">
                    </div>
                    <div class="input-container">
                        <input type="date" name="date" class="input">
                    </div>
                    <div class="input-container">
                        <div class="input">
                            <label for="from">From: </label>
                            <select name="starting" id="from" name="starting">
                                <option value="6:00-7:00">6:00-7:00</option>
                                <option value="7:00-8:00">7:00-8:00</option>
                                <option value="8:00-9:00">8:00-9:00</option>
                                <option value="9:00-10:00">9:00-10:00</option>
                                <!-- Add more time options if needed -->
                            </select>
                        </div>
                    </div>
                    <input type="submit" value="Book Now" class="send" />
                </form>
                <p><?php echo $message; ?></p>
            </div>
        </div>
    </section>

    <div class="footer">
        <div class="inner-footer">
            <div class="footer-items">
                <h1>Website Name</h1>
                <p>Kick Vision AI</p>
            </div>
            <div class="footer-items">
                <h3>Quick Links</h3>
                <div class="border1"></div>
                <ul>
                    <a href="homepage.php"><li>Home</li></a>
                    <a href="contactus.php"><li>Contact</li></a>
                    <a href="index.php#about"><li>About</li></a>
                </ul>
            </div>
            <div class="footer-items">
                <h3>Services</h3>
                <div class="border1"></div>
                <ul>
                    <a href="homepage.php#futsals"><li>futsal Booking</li></a>
                    <!-- Add more service links if needed -->
                </ul>
            </div>
            <div class="footer-items">
                <h3>Contact us</h3>
                <div class="border1"></div>
                <ul>
                    <li><i class="fa fa-map-marker" aria-hidden="true"></i>Dillibazar, KTM</li>
                    <li><i class="fa fa-phone" aria-hidden="true"></i>123456789</li>
                    <li><i class="fa fa-envelope" aria-hidden="true"></i>KickVisionAi@gmail.com</li>
                </ul>
                <div class="social-media">
                    <a href="https://www.instagram.com/"><i class="fab fa-instagram"></i></a>
                    <a href="https://www.facebook.com/"><i class="fab fa-facebook"></i></a>
                    <a href="https://www.google.com/"><i class="fab fa-google-plus-square"></i></a>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/swiper@8/swiper-bundle.min.js"></script>
    <script>
        const swiper = new Swiper('.swiper', {
            autoplay: {
                delay: 5000,
                disableOnInteraction: false
            },
            loop: true,
            // Add more swiper options if needed
        });
    </script>
</body>

</html>
