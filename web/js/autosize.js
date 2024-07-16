import { app } from '../../../scripts/app.js'
app.registerExtension({
    name: "AdvancedControlNet.autosize",
    async nodeCreated(node) {
        if(node.acnAutosize) {
            let size = node.computeSize(0);
            size[0] += node.acnAutosize?.padding || 0;
            node.setSize(size);
        }
    },
    async getCustomWidgets() {
        return {
            ACNAUTOSIZE(node, inputName, inputData) {
                let w = {
                    name : inputName,
                    type : "ACN.AUTOSIZE",
                    value : "",
                    options : {"serialize": false},
                    computeSize : function(width) {
                        return [0, -4];
                    }
                }
                node.acnAutosize = inputData[1];
                if (!node.widgets) {
                    node.widgets = []
                }
                node.widgets.push(w)
                return w;
            }
        }
    }
});
