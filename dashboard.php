<?php
include 'db.php';

session_start();
if (isset($_SESSION['user'])) {
    if ($_SESSION["user"]['user_email'] != "admin@admin.com") {
        header('Location: dashboard.php');
    }
} else {
    header('Location: dashboard.php');
}

// Fetch data from the bookedfutsal table
$sql = "SELECT * FROM bookedfutsal AS b 
        INNER JOIN signup AS s ON b.user_id = s.id 
        INNER JOIN futsal AS f ON b.futsal_id = f.id";
$result = mysqli_query($conn, $sql);

// Check if the query was successful
if ($result && mysqli_num_rows($result) > 0) {
    $bookings = mysqli_fetch_all($result, MYSQLI_ASSOC);
} else {
    $bookings = []; // Set an empty array if no bookings found
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
  <link rel="stylesheet" href="css/dashboard.css">

  <title></title>
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
  <table class="table">
    <thead class="thead-dark">
      <tr>
        <th scope="col">Id</th>
        <th scope="col">Name</th>
        <th scope="col">Phone Number</th>
        <th scope="col">Futsal Name</th>
        <th scope="col">Booked Date</th>
        <th scope="col">Time</th>
        <th scope="col">Action</th>
      </tr>
    </thead>
    <tbody>
      <?php foreach ($bookings as $booking) : ?>
        <tr>
          <td><?php echo $booking['id']; ?></td>
          <td><?php echo $booking['user_name']; ?></td>
          <td><?php echo $booking['user_phone']; ?></td>
          <td><?php echo $booking['hname']; ?></td>
          <td><?php echo $booking['date']; ?></td>
          <td><?php echo $booking['starting_time']; ?></td>
          <td>
            <form method="post" action="/admin/changepassword" object="<?php echo $booking['id']; ?>">
              <input type="hidden" name="email" value="<?php echo $booking['user_email']; ?>">
              <button type="submit" class="btn btn-primary">Contact</button>
            </form>
          </td>
        </tr>
      <?php endforeach; ?>
    </tbody>
  </table>
</div>


</body>

</html>
