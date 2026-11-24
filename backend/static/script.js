// -----------------------------
// GOOGLE MAP
// -----------------------------

function initMap() {

    const center = {
        lat: 12.9716,
        lng: 77.5946
    }

    const map = new google.maps.Map(
        document.getElementById("map"),
        {
            zoom: 12,
            center: center
        }
    )

    // Main Property Marker

    new google.maps.Marker({
        position: center,
        map: map,
        title: "Property Location"
    })

    // Nearby Places

    const places = [
        {
            lat: 12.975,
            lng: 77.59,
            title: "Hospital"
        },
        {
            lat: 12.968,
            lng: 77.60,
            title: "School"
        },
        {
            lat: 12.972,
            lng: 77.61,
            title: "Mall"
        }
    ]

    places.forEach(place => {

        new google.maps.Marker({
            position: {
                lat: place.lat,
                lng: place.lng
            },
            map: map,
            title: place.title
        })
    })
}

// -----------------------------
// LOAD MAP
// -----------------------------

window.onload = initMap

