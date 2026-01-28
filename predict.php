<?php
include "db.php";

session_start();
if (!isset($_SESSION['user'])) {
    header("Location: login.php");
    exit;
}
?>

<!DOCTYPE html>

<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Predict page</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta2/css/all.min.css" integrity="sha512-YWzhKL2whUzgiheMoBFwW8CKV4qpHQAEuvilg9FAn5VJUDwKZZxkJNuGM4XkWuk94WCrrwslk8yWNGmY1EduTA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@8/swiper-bundle.min.css" />
    <link rel="stylesheet" href="css/afterlogins.css" />
    <link rel="stylesheet" href="css/pred.css" />
    <style>
    button {
        padding: 8px 16px;
        font-size: 14px;
        background-color:rgb(15, 101, 206);
        color: white;
        border: none;
        border-radius: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }

    button:hover {
        background-color:rgb(26, 169, 45);
    }

    button:focus {
        outline: none;
    }

    /* Loader Animation */
    .loader-container {
        display: none;
        justify-content: center;
        align-items: center;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.7);
        z-index: 1000;
        flex-direction: column;
        color: white;
        font-size: 20px;
    }

    .loader {
        border: 6px solid #f3f3f3;
        border-top: 6px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin-bottom: 10px;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
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
            <a href="Homepage.php#futsals">Futsal</a>
            <a href="index.php#review">services</a>
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
                <div class="swiper-slide"><img src="images/hos4.jpg" alt=""></div>
                <div class="swiper-slide"><img src="images/hos2.jpeg" alt=""></div>
                <div class="swiper-slide"><img src="images/hos5.jpg" alt=""></div>
            </div>
        </div>
    </div>

    <section id="contact-us">
        <span class="big-circle"></span>
        <div class="form">
            <div class="contact-info">
                <h3 class="heading">Analyze Here</h3>
                <img src="images/ucl1.png" alt="Futsal" srcset="" />
            </div>
            <div class="contact-form">
                <form method="post" action="predict2.php" enctype="multipart/form-data" onsubmit="showLoader()">
                    <span class="circle one"></span>
                    <span class="circle two"></span>
                    <h3 class="heading"><span style="color: white;">Select</span> <span>Demo</span> <span>Video</span></h3>
                    <div class="demo-video-selection" style="font-size: 20px;">
                        <label style="display: inline-block; margin-right: 20px; color: white;">
            <input type="radio" name="demo_video" value="demo1.mp4" style="transform: scale(1.5); "> Demo 1
        </label>
        <label style="display: inline-block; margin-right: 20px; color: white;">
            <input type="radio" name="demo_video" value="demo2.mp4" style="transform: scale(1.5); "> Demo 2
        </label>
        <label style="display: inline-block; margin-right: 20px; color: white;">
            <input type="radio" name="demo_video" value="demo3.mp4" style="transform: scale(1.5); "> Demo 3
        </label>
                    </div>
                    <h3 class="heading"><span style="color: white;">Browse</span> <span>Video</span></h3>
                    <div class="input-container">
                        <input type="text" id="file-name" class="input" placeholder="No file selected" readonly>
                        <input type="file" id="file-input" name="file" style="display: none;">
                        <button type="button" onclick="document.getElementById('file-input').click();">Browse</button>
                    </div>
                    <input type="submit" value="Analyze" class="send" />
                </form>
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
                    <a href="homepage.php#futsals"><li>Futsal Booking</li></a>
                    <!-- Add more service links if needed -->
                </ul>
            </div>
            <div class="footer-items">
                <h3>Contact us</h3>
                <div class="border1"></div>
                <ul>
                    <li><i class="fa fa-map-marker" aria-hidden="true"></i>Dillibazar, KTM</li>
                    <li><i class="fa fa-phone" aria-hidden="true"></i>123456789</li>
                    <li><i class="fa fa-envelope" aria-hidden="true"></i>kickvisionai@gmail.com</li>
                </ul>
                <div class="social-media">
                    <a href="https://www.instagram.com/"><i class="fab fa-instagram"></i></a>
                    <a href="https://www.facebook.com/"><i class="fab fa-facebook"></i></a>
                    <a href="https://www.google.com/"><i class="fab fa-google-plus-square"></i></a>
                </div>
            </div>
        </div>
    </div>
    <div class="loader-container" id="loader">
        <div class="loader"></div>
        <p>Processing Video...</p>
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
   <script>
    document.getElementById('file-input').addEventListener('change', function() {
        var fileName = this.files[0] ? this.files[0].name : "No file selected";
        document.getElementById('file-name').value = fileName;
    });
</script>
    <script>
        function showLoader() {
            document.getElementById('loader').style.display = 'flex';
        }
    </script>
</body>
</html>
