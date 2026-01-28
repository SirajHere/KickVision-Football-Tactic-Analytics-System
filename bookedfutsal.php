<?php
include "db.php";

session_start();
if (!isset($_SESSION['user'])) {
    header("Location: login.php");
}

if (isset($_GET['action']) && $_GET['action'] === 'cancel' && isset($_GET['booking_id'])) {
    // Get the booking ID from the URL
    $bookingId = $_GET['booking_id'];

    // Delete the booking record from the database
    $deleteQuery = "DELETE FROM bookedfutsal WHERE id = '$bookingId'";
    if (mysqli_query($conn, $deleteQuery)) {
        // Redirect to the same page after successful booking cancellation
        header("Location: bookedfutsal.php");
        exit();
    } else {
        echo "Error: " . mysqli_error($conn);
    }
}

// Fetch the booked futsals for the logged-in user
$userId = $_SESSION['user']['id'];
$selectQuery = "SELECT bf.id AS bookId, s.user_name AS userName, f.hname AS futsalName, bf.date, bf.starting_time
                FROM bookedfutsal bf
                INNER JOIN signup s ON bf.user_id = s.id
                INNER JOIN futsal f ON bf.futsal_id = f.id
                WHERE bf.user_id = '$userId'";

$result = mysqli_query($conn, $selectQuery);
$books = mysqli_fetch_all($result, MYSQLI_ASSOC);
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css" />
    <link rel="stylesheet" href="css/bookedhsopital.css">
    <title>Booked Futsal</title>
</head>

<body>
<div class="backbuttton">
    <a href="homepage.php"><button type="button">Go Back</button></a>
</div>
<div class="innercontent">
    <h3>Here Are Your Booked Futsals</h3>
    <table class="table">
        <thead class="thead-dark">
        <tr>
            <th scope="col">Id</th>
            <th scope="col">Name</th>
            <th scope="col">Futsal Name</th>
            <th scope="col">Booked Date</th>
            <th scope="col">Time</th>
            <th scope="col">Action</th>
        </tr>
        </thead>
        <tbody>
        <?php foreach ($books as $booking) { ?>
            <tr>
                <td><?php echo $booking['bookId']; ?></td>
                <td><?php echo $booking['userName']; ?></td>
                <td><?php echo $booking['futsalName']; ?></td>
                <td><?php echo $booking['date']; ?></td>
                <td><?php echo $booking['starting_time']; ?></td>
                <td>
                    <a href="bookedfutsal.php?action=cancel&booking_id=<?php echo $booking['bookId']; ?>">
                        <button type="button" class="btn btn-danger">Cancel Booking</button>
                    </a>
                </td>
            </tr>
        <?php } ?>
        </tbody>
    </table>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta3/dist/js/bootstrap.bundle.min.js"></script>
</body>

</html>
