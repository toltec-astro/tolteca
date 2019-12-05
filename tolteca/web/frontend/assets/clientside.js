if (!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.tolteca = {
    array_size: function(a) {
        if (a) {
            return a.length
        }
        return null
    },
    array_concat: function(a, b) {
        if (!a) {
            return b
        }
        return a.concat(b)
    },
    array_first_id: function(a) {
        if (a) {
            return a[0]['id'];
        }
        return null
    }
}
