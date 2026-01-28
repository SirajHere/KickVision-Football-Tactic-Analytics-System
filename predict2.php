<?php
include "db.php";

session_start();
if (!isset($_SESSION['user'])) {
    header("Location: login.php");
    exit;
}
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Predicted Page</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta2/css/all.min.css" integrity="sha512-YWzhKL2whUzgiheMoBFwW8CKV4qpHQAEuvilg9FAn5VJUDwKZZxkJNuGM4XkWuk94WCrrwslk8yWNGmY1EduTA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@8/swiper-bundle.min.css" />
    <link rel="stylesheet" href="css/afterlogins.css" />
    <link rel="stylesheet" href="css/output.css" />
    <style>
        .video-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    margin-top: 120px;
     /* Increased from 50px to 150px */
    width: 240%; /* Triple the size */
    margin-left: -15%
}

.video-container video {
    width: 80%; /* Adjust to fit the container */
    max-width: 2600px; /* Increased from 800px to 2400px */
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
}

    </style>
</head>

<body>
    <header>
        <a href="Homepage.html" class="logo">
            <img src="images/ucl1.png" alt="Futsal" srcset="" />Kick Vision AI
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
                <h3 class="heading">Analyzed video</h3>
                <?php
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (!empty($_POST["demo_video"])) {
        // A demo video was selected, send only its name
        $selectedDemoVideo = $_POST["demo_video"];
        
        // Send demo video name to Flask for processing
        $response = sendDemoToFlask($selectedDemoVideo);

        if ($response) {
            // Assuming Flask might return the video in AVI format, let's handle the conversion
            $aviFilePath = 'demo_video.avi';
            file_put_contents($aviFilePath, $response);

            // Convert the AVI to MP4
            $mp4FilePath = 'demo_video.mp4';
            $ffmpegCommand = "ffmpeg -y -i $aviFilePath -c:v libx264 -c:a aac -strict experimental $mp4FilePath";
            exec($ffmpegCommand);

            // Display the processed and converted video
            echo "<h2></h2>";
            echo "<div class='video-container'>
                    <video controls>
                        <source src='$mp4FilePath?" . time() . "' type='video/mp4'>
                        Your browser does not support the video tag.
                    </video>
                  </div>";

            unlink($aviFilePath); // Clean up the temporary AVI file
        } else {
            echo "<p>Error: Flask did not return a valid response for demo video.</p>";
        }
    } elseif (isset($_FILES['file']) && $_FILES['file']['size'] > 0) {
        // A file was uploaded, process it
        $videoFile = $_FILES['file'];

        // Ensure the uploads directory exists
        $uploadDirectory = "uploads/";
        if (!is_dir($uploadDirectory)) {
            mkdir($uploadDirectory, 0777, true);
        }

        // Generate a unique filename
        $uploadedFilePath = $uploadDirectory . uniqid("video_", true) . "_" . basename($videoFile["name"]);
        if (move_uploaded_file($videoFile["tmp_name"], $uploadedFilePath)) {
            // Send the video file to Flask for processing
            $processedVideo = sendToFlask($uploadedFilePath);

            if ($processedVideo) {
                // Save and convert to MP4
                $aviFilePath = 'processed_video.avi';
                file_put_contents($aviFilePath, $processedVideo);

                $mp4FilePath = 'processed_video.mp4';
                $ffmpegCommand = "ffmpeg -y -i $aviFilePath -c:v libx264 -c:a aac -strict experimental $mp4FilePath";
                exec($ffmpegCommand);

                echo  "<h2></h2>";
                echo "<div class='video-container'>
                        <video controls>
                            <source src='$mp4FilePath?" . time() . "' type='video/mp4'>
                            Your browser does not support the video tag.
                        </video>
                      </div>";

                unlink($aviFilePath); // Clean up the temporary AVI file
            } else {
                echo "<p>Error: Flask did not return a valid response.</p>";
            }
        } else {
            echo "<p>Error: Failed to upload the file.</p>";
        }
    } else {
        echo "<p>Error: No file uploaded or demo video selected.</p>";
    }
}

// Function to send the uploaded video to Flask
function sendToFlask($videoFilePath) {
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, 'http://127.0.0.1:5000/process_video'); 
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);

    $cfile = curl_file_create($videoFilePath, 'video/mp4', basename($videoFilePath));
    $data = array('file' => $cfile);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);

    $response = curl_exec($ch);
    if ($response === false) {
        echo "<p>cURL Error: " . curl_error($ch) . "</p>";
    }

    curl_close($ch);
    return $response;
}

// Function to send the demo video name to Flask
function sendDemoToFlask($demoVideo) {
    $ch = curl_init();
    curl_setopt($ch, CURLOPT_URL, 'http://127.0.0.1:5000/process_demo_video'); 
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);

    $data = array('demo_video' => $demoVideo);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $data);

    $response = curl_exec($ch);
    if ($response === false) {
        echo "<p>cURL Error: " . curl_error($ch) . "</p>";
    }

    curl_close($ch);
    return $response;
}
?>


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
                </ul>
            </div>
            <div class="footer-items">
                <h3>Contact us</h3>
                <div class="border1"></div>
                <ul>
                    <li><i class="fa fa-map-marker" aria-hidden="true"></i>Dillibazar, KTM</li>
                    <li><i class="fa fa-phone" aria-hidden="true"></i>123456789</li>
                    <li><i class="fa fa-envelope" aria-hidden="true"></i>Kickvisionai@gmail.com</li>
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
        });
    </script>

</body>

</html>
