if (!window.dash_clientside) {window.dash_clientside = {};}
window.dash_clientside.tolteca = {
    interface_from_latest_data: function(changed, data, interface_options) {
        console.log(changed)
        if (!changed || !data) {
            return interface_options
        }
        options = [];
        data[0]['RoachIndex'].split(',').forEach(
            e => options.push({
                'label': "toltec" + e,
                'value': "toltec" + e
                })
        )
        return options
    },
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
        if (!a || a.length == 0) {
            // console.log("no update")
            return b
        }
        return a.concat(b)
    },
}


// https://community.plot.ly/t/links-in-datatable-multipage-app/26081/6
window.dash_clientside.ui = {
    collapseWithClick: function(n, classname) {
        console.log(n, classname)
        if (n) {
            if (classname && classname.includes(" collapsed")) {
                return classname.replace(" collapsed", "")
            }
            return classname + " collapsed"
        }
        return classname
    },
    toggleWithClick: function(n, is_open) {
        if (n)
            return !is_open
        return is_open
    },
    replaceWithLinks: function(trigger, table_id) {
        let cells = document.getElementById(table_id)
            .getElementsByClassName("dash-cell column-1");
        base_route = "/"

        cells.forEach((elem, index, array) => {
            elem.children[0].innerHTML =
                '<a href="' +
                base_route +
                elem.children[0].innerText +
                '" target="_some_target" rel="noopener noreferrer">' +
                elem.children[0].innerText +
                "</a>";
        });
        return null;
    }
}
