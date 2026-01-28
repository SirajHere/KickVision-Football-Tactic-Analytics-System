<?php
include 'db.php';

session_start();
if (isset($_SESSION['user'])) {
    if ($_SESSION["user"]['user_email'] != "admin@admin.com") {
        header('Location: homepage.php');
    }
} else {
    header('Location: homepage.php');
}

if (isset($_GET['delete_id'])) {
    $delete_id = $_GET['delete_id'];

    $sql = "DELETE FROM contact WHERE id = $delete_id";

    if (mysqli_query($conn, $sql)) {
        header('Location: viewreview.php');
        exit();
    } else {
        echo "Error deleting entry: " . mysqli_error($conn);
    }

    mysqli_close($conn);
}

$sql = "SELECT * FROM contact";
$result = mysqli_query($conn, $sql);

?>

<!DOCTYPE html>
<html lang="en">


<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" />
    <link rel="stylesheet" href="css/viewreview.css">

    <title>View Reviews</title>
</head>

<body>
    <!-- sidebar content from here -->
    <div class="sidebarcontent">
        <h3>Admin Panel</h3>

        <a href="dashboard.php"> <i class="fas fa-tachometer-alt me-2">Dashboard</i></a>
        <a href="addfutsal.php"><i class="fas fa-project-diagram me-2">Futsal</i></a>
        <a href="viewfutsal.php"><i class="fas fa-chart-line me-2">View Futsal</i> </a>
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


</div>>

    <div class="innercontent">
        <h2>View Reviews</h2>
        <table class="table">
            <thead class="thead-dark">
                <tr>
                    <th scope="col">Id</th>
                    <th scope="col">Name</th>
                    <th scope="col">Email</th>
                    <th scope="col">Subject</th>
                    <th scope="col">Message</th>
                    <th scope="col">Action</th>
                </tr>
            </thead>
            <tbody>
                <?php while ($row = mysqli_fetch_assoc($result)) : ?>
                    <tr>
                        <td><?php echo $row['id']; ?></td>
                        <td><?php echo $row['contactname']; ?></td>
                        <td><?php echo $row['contactemail']; ?></td>
                        <td><?php echo $row['contactsubject']; ?></td>
                        <td><?php echo $row['contactmessage']; ?></td>
                        <td>
                            <a href="viewreview.php?delete_id=<?php echo $row['id']; ?>" class="btn btn-danger" onclick="return confirm('Are you sure you want to delete this entry?')">Delete</a>
                        </td>
                    </tr>
                <?php endwhile; ?>
            </tbody>
        </table>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
