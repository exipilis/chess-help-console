<?
$img = $_POST['img'];
$fn = $_POST['fn'];

$data = base64_decode(preg_replace('#^data:image/\w+;base64,#i', '', $img));
file_put_contents('pngs/board_' . $fn . '.png', $data);

echo sizeof($data);

?>