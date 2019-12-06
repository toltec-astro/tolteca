if (!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.tolteca = {
    array_summary: function(a, s) {
        // console.log("summary")
        // console.log(a)
        // console.log(s)
        if (a) {
            return {
                ...s,
                'size': a.length,
                'first': a[0]
                }
        }
        return s
    },
    array_concat: function(a, b) {
        // console.log("concat")
        // console.log(a)
        // console.log(b)
        if (!a) {
            return b
        }
        return a.concat(b)
    },
}
