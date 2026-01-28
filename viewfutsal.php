<?php
include 'db.php';

session_start();
if (isset($_SESSION['user'])) {
    if ($_SESSION["user"]['user_email'] != "admin@admin.com") {
        header('Location: dashboard.php');
    }
} else {
    header('Location:dashboard.php');
}

// Fetch futsal data from the database
$sql = "SELECT * FROM futsal";
$result = mysqli_query($conn, $sql);

// Create an array to store the futsal data
$futsals = [];
while ($row = mysqli_fetch_assoc($result)) {
    $futsals[] = $row;
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" />
    <link rel="stylesheet" href="css/viewfutsals.css">
    <title>Admin Panel - View Futsal</title>
</head>
<body>
    <!-- sidebar content from here -->
    <div class="sidebarcontent">
  <h3>Admin Panel</h3>


  <a href="dashboard.php"><i class="fas fa-tachometer-alt me-2">Dashboard</i></a>
  <a href="addfutsal.php"><i class="fas fa-project-diagram me-2">Futsal</i></a>
  <a href="viewfutsal.php"> <i class="fas fa-chart-line me-2">View Futsal</i></a>
  <a href="viewreview.php"><i class="fas fa-chart-line me-2">Futsal Review</i></a>
  <a href="viewreview.php"><i class="fas fa-paperclip me-2">Reports</i></a>
  
  <div class="logout">
    <i class="fas fa-power-off me-2" href="logout.php">Logout</i>
    
  </div>
  <script>
document.querySelector(".logout").addEventListener("click", function() {
   
    window.location.href = "logout.php";
});
</script>


</div>

    <div class="innercontent">
        <?php foreach ($futsals as $futsal) : ?>
            <div class="indiviudal-futsal">
                <a href="#">
                    <img src="<?php echo $futsal['image3']; ?>" alt="<?php echo $futsal['hname']; ?>">
                    <h2><?php echo $futsal['hname']; ?></h2>
                    <p><?php echo $futsal['description']; ?></p>
                </a>
            </div>
        <?php endforeach; ?>
    </div>
</body>
</html>
