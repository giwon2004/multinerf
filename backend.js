window.onload = function(){
    var objembedobj='cube';
    var encodeobj='cube';
    var onehotobj = 'cube';
    var kerasobj = 'cube';
    var leftexp = 'onehot';
    var rightexp = 'onehot';
    var leftiter = '3000';
    var rightiter = '3000';
    var overallobj='cube';
    var objembedplay = true;
    var encodeplay = true;
    var encodeplay2 = true;

    function changebtncolor(btnclass,btnnum){
        $(btnclass).children('#3000').css('background-color','#eee');
        $(btnclass).children('#5000').css('background-color','#eee');
        $(btnclass).children('#10000').css('background-color','#eee');
        $(btnclass).children('#50000').css('background-color','#eee');
        $(btnclass).children('#gt').css('background-color','#eee');
        $(btnclass).children(btnnum).css('background-color','skyblue');
    }

    function changeimgsrc(classname,btnname,obj){
        $(classname).children(btnname).click(function(){
            if(obj == 'cube'){
                
            }
            else if(obj == 'greek'){

            }
        });
    }

    function srcoverallimg(exp,iter,obj){
        var src = "";
        src += "img/";
        src += exp + "/";
        src += obj;
        src += iter+".png";
        if(iter == "gt")
        {
            src = "img/" + "gt" + obj + ".png";
        }
        return src;
    }

    //overall 제어
    $(".overallbtnleft").children('#selectexp').on("change",function(){
        leftexp = $(this).find("option:selected").val();
        overallmodelimg.src = srcoverallimg(leftexp,leftiter,overallobj);
    })
    $(".overallbtnright").children('#selectexp').on("change",function(){
        rightexp = $(this).find("option:selected").val();
        overallgtimg.src = srcoverallimg(rightexp,rightiter,overallobj);
    })
    $(".overallinput").children('#selectobj').on("change",function(){
        overallobj = $(this).find("option:selected").val();
    })

    $(".overallbtnleft").children('#3000').click(function(){
        leftiter = "3000";
        overallmodelimg.src = srcoverallimg(leftexp,leftiter,overallobj);
        changebtncolor(".overallbtnleft",'#3000');
    });
    $(".overallbtnleft").children('#5000').click(function(){
        leftiter = "5000";
        overallmodelimg.src = srcoverallimg(leftexp,leftiter,overallobj);
        changebtncolor(".overallbtnleft",'#5000');
    });
    $(".overallbtnleft").children('#10000').click(function(){
        leftiter = "10000";
        overallmodelimg.src = srcoverallimg(leftexp,leftiter,overallobj);
        changebtncolor(".overallbtnleft",'#10000');
    });
    $(".overallbtnleft").children('#50000').click(function(){
        leftiter = "50000";
        overallmodelimg.src = srcoverallimg(leftexp,leftiter,overallobj);
        changebtncolor(".overallbtnleft",'#50000');
    });
    $(".overallbtnleft").children('#gt').click(function(){
        leftiter = "gt";
        overallmodelimg.src = srcoverallimg(leftexp,leftiter,overallobj);
        changebtncolor(".overallbtnleft",'#gt');
    });

    $(".overallbtnright").children('#3000').click(function(){
        rightiter = "3000";
        overallgtimg.src = srcoverallimg(rightexp,rightiter,overallobj);
        changebtncolor(".overallbtnright",'#3000');
    });
    $(".overallbtnright").children('#5000').click(function(){
        rightiter = "5000";
        overallgtimg.src = srcoverallimg(rightexp,rightiter,overallobj);
        changebtncolor(".overallbtnright",'#5000');
    });
    $(".overallbtnright").children('#10000').click(function(){
        rightiter = "10000";
        overallgtimg.src = srcoverallimg(rightexp,rightiter,overallobj);
        changebtncolor(".overallbtnright",'#10000');
    });
    $(".overallbtnright").children('#50000').click(function(){
        rightiter = "50000";
        overallgtimg.src = srcoverallimg(rightexp,rightiter,overallobj);
        changebtncolor(".overallbtnright",'#50000');
    });
    $(".overallbtnright").children('#gt').click(function(){
        rightiter = "gt";
        overallgtimg.src = srcoverallimg(rightexp,rightiter,overallobj);
        changebtncolor(".overallbtnright",'#gt');
    });

    //onehot 제어
    $(".onehotbtnleft").children('#3000').click(function(){
        if(onehotobj == 'cube'){
            onehotmodelimg.src = "img/onehot/cube3000.png";
            changebtncolor(".onehotbtnleft",'#3000');
        }
        else if(onehotobj == 'greek'){
            onehotmodelimg.src = "img/onehot/greek3000.png";
            changebtncolor(".onehotbtnleft",'#3000');
        }
    });
    $(".onehotbtnleft").children('#5000').click(function(){
        if(onehotobj == 'cube'){
            onehotmodelimg.src = "img/onehot/cube5000.png";
            changebtncolor(".onehotbtnleft",'#5000');
        }
        else if(onehotobj == 'greek'){
            onehotmodelimg.src = "img/onehot/greek5000.png";
            changebtncolor(".onehotbtnleft",'#5000');
        }
    });
    $(".onehotbtnleft").children('#10000').click(function(){
        if(onehotobj == 'cube'){
            onehotmodelimg.src = "img/onehot/cube10000.png";
            changebtncolor(".onehotbtnleft",'#10000');
        }
        else if(onehotobj == 'greek'){
            onehotmodelimg.src = "img/onehot/greek10000.png";
            changebtncolor(".onehotbtnleft",'#10000');
        }
    });
    $(".onehotbtnleft").children('#50000').click(function(){
        if(onehotobj == 'cube'){
            onehotmodelimg.src = "img/onehot/cube50000.png";
            changebtncolor(".onehotbtnleft",'#50000');
        }
        else if(onehotobj == 'greek'){
            onehotmodelimg.src = "img/onehot/greek50000.png";
            changebtncolor(".onehotbtnleft",'#50000');
        }
    });
    $(".onehotbtnleft").children('#gt').click(function(){
        if(onehotobj == 'cube'){
            onehotmodelimg.src = "img/gtcube.png";
            changebtncolor(".onehotbtnleft",'#gt');
        }
        else if(onehotobj == 'greek'){
            onehotmodelimg.src = "img/gtgreek.png";
            changebtncolor(".onehotbtnleft",'#gt');
        }
    });

    $(".onehotbtnright").children('#3000').click(function(){
        if(onehotobj == 'cube'){
            onehotgtimg.src = "img/onehot/cube3000.png";
            changebtncolor(".onehotbtnright",'#3000');
        }
        else if(onehotobj == 'greek'){
            onehotgtimg.src = "img/onehot/greek3000.png";
            changebtncolor(".onehotbtnright",'#3000');
        }
    });
    $(".onehotbtnright").children('#5000').click(function(){
        if(onehotobj == 'cube'){
            onehotgtimg.src = "img/onehot/cube5000.png";
            changebtncolor(".onehotbtnright",'#5000');
        }
        else if(onehotobj == 'greek'){
            onehotgtimg.src = "img/onehot/greek5000.png";
            changebtncolor(".onehotbtnright",'#5000');
        }
    });
    $(".onehotbtnright").children('#10000').click(function(){
        if(onehotobj == 'cube'){
            onehotgtimg.src = "img/onehot/cube10000.png";
            changebtncolor(".onehotbtnright",'#10000');
        }
        else if(onehotobj == 'greek'){
            onehotgtimg.src = "img/onehot/greek10000.png";
            changebtncolor(".onehotbtnright",'#10000');
        }
    });
    $(".onehotbtnright").children('#50000').click(function(){
        if(onehotobj == 'cube'){
            onehotgtimg.src = "img/onehot/cube50000.png";
            changebtncolor(".onehotbtnright",'#50000');
        }
        else if(onehotobj == 'greek'){
            onehotgtimg.src = "img/onehot/greek50000.png";
            changebtncolor(".onehotbtnright",'#50000');
        }
    });
    $(".onehotbtnright").children('#gt').click(function(){
        if(onehotobj == 'cube'){
            onehotgtimg.src = "img/gtcube.png";
            changebtncolor(".onehotbtnright",'#gt');
        }
        else if(onehotobj == 'greek'){
            onehotgtimg.src = "img/gtgreek.png";
            changebtncolor(".onehotbtnright",'#gt');
        }
    });
    $(".onehotinput").children('#selectobj').on("change", function(){
        onehotobj = $(this).find("option:selected").val();
    })


    // objembed 제어
    $(".objembedbtnleft").children('#3000').click(function(){
        if(objembedobj == 'cube'){
            //gtimg.src = "img/gtcube3000.png";
            modelimg.src = "img/objembed/cube3000.png";
            changebtncolor(".objembedbtnleft",'#3000');
        }
        else if(objembedobj == 'greek'){
            //gtimg.src = "img/gtgreek3000.png";
            modelimg.src = "img/objembed/greek3000.png";
            changebtncolor(".objembedbtnleft",'#3000');
        }
    });
    $(".objembedbtnleft").children('#5000').click(function(){
        if(objembedobj == 'cube'){
            //gtimg.src = "img/gtcube5000.png";
            modelimg.src = "img/objembed/cube5000.png";
            changebtncolor(".objembedbtnleft",'#5000');
        }
        else if(objembedobj == 'greek'){
            //gtimg.src = "img/gtgreek5000.png";
            modelimg.src = "img/objembed/greek5000.png";
            changebtncolor(".objembedbtnleft",'#5000');
        }
    });
    $(".objembedbtnleft").children('#10000').click(function(){
        if(objembedobj == 'cube'){
            //gtimg.src = "img/gtcube10000.png";
            modelimg.src = "img/objembed/cube10000.png";
            changebtncolor(".objembedbtnleft",'#10000');
        }
        else if(objembedobj == 'greek'){
            //gtimg.src = "img/gtgreek10000.png";
            modelimg.src = "img/objembed/greek10000.png";
            changebtncolor(".objembedbtnleft",'#10000');
        }
    });
    $(".objembedbtnleft").children('#50000').click(function(){
        if(objembedobj == 'cube'){
            //gtimg.src = "img/gtcube50000.png";
            modelimg.src = "img/objembed/cube50000.png";
            changebtncolor(".objembedbtnleft",'#50000');
        }
        else if(objembedobj == 'greek'){
            //gtimg.src = "img/gtgreek50000.png";
            modelimg.src = "img/objembed/greek50000.png";
            changebtncolor(".objembedbtnleft",'#50000');
        }
    })
    $(".objembedbtnleft").children('#gt').click(function(){
        if(objembedobj == 'cube'){
            //gtimg.src = "img/gtcube50000.png";
            modelimg.src = "img/objembed/gtcube.png";
            changebtncolor(".objembedbtnleft",'#gt');
        }
        else if(objembedobj == 'greek'){
            //gtimg.src = "img/gtgreek50000.png";
            modelimg.src = "img/objembed/gtgreek.png";
            changebtncolor(".objembedbtnleft",'#gt');
        }
    })


    $(".objembedbtnright").children('#3000').click(function(){
        if(objembedobj == 'cube'){
            gtimg.src = "img/objembed/cube3000.png";
            //modelimg.src = "img/cube003000.png";
            changebtncolor(".objembedbtnright",'#3000');
        }
        else if(objembedobj == 'greek'){
            gtimg.src = "img/objembed/greek3000.png";
            //modelimg.src = "img/greek003000.png";
            changebtncolor(".objembedbtnright",'#3000');
        }
    });
    $(".objembedbtnright").children('#5000').click(function(){
        if(objembedobj == 'cube'){
            gtimg.src = "img/objembed/cube5000.png";
            //modelimg.src = "img/cube005000.png";
            changebtncolor(".objembedbtnright",'#5000');
        }
        else if(objembedobj == 'greek'){
            gtimg.src = "img/objembed/greek5000.png";
            //modelimg.src = "img/greek005000.png";
            changebtncolor(".objembedbtnright",'#5000');
        }
    });
    $(".objembedbtnright").children('#10000').click(function(){
        if(objembedobj == 'cube'){
            gtimg.src = "img/objembed/cube10000.png";
            //modelimg.src = "img/cube010000.png";
            changebtncolor(".objembedbtnright",'#10000');
        }
        else if(objembedobj == 'greek'){
            gtimg.src = "img/objembed/greek10000.png";
            //modelimg.src = "img/greek010000.png";
            changebtncolor(".objembedbtnright",'#10000');
        }
    });
    $(".objembedbtnright").children('#50000').click(function(){
        if(objembedobj == 'cube'){
            gtimg.src = "img/objembed/cube50000.png";
            //modelimg.src = "img/cube050000.png";
            changebtncolor(".objembedbtnright",'#50000');
        }
        else if(objembedobj == 'greek'){
            gtimg.src = "img/objembed/greek50000.png";
            //modelimg.src = "img/greek050000.png";
            changebtncolor(".objembedbtnright",'#50000');
        }
    })
    $(".objembedbtnright").children('#gt').click(function(){
        if(objembedobj == 'cube'){
            gtimg.src = "img/objembed/gtcube.png";
            //modelimg.src = "img/gtcube.png";
            changebtncolor(".objembedbtnright",'#gt');
        }
        else if(objembedobj == 'greek'){
            gtimg.src = "img/objembed/gtgreek.png";
           // modelimg.src = "img/gtgreek.png";
           changebtncolor(".objembedbtnright",'#gt');
        }
    })
    
    $(".objembedinput").children('#selectobj').on("change", function(){
        objembedobj = $(this).find("option:selected").val();
    })
    $('.objembedvideodiv').children('.video1').click(function(){
        if(objembedplay == false){
            $('.objembedvideodiv').children('.video2').get(0).play();
            $('.objembedvideodiv').children('.video1').get(0).play();
            objembedplay = true;
        }
        else if(objembedplay == true){
            $('.objembedvideodiv').children('.video2').get(0).pause();
            $('.objembedvideodiv').children('.video1').get(0).pause();
            objembedplay = false;
        }
    })
    $('.objembedvideodiv').children('.video2').click(function(){
        if(objembedplay == false){
            $('.objembedvideodiv').children('.video2').get(0).play();
            $('.objembedvideodiv').children('.video1').get(0).play();
            objembedplay = true;
        }
        else if(objembedplay == true){
            $('.objembedvideodiv').children('.video2').get(0).pause();
            $('.objembedvideodiv').children('.video1').get(0).pause();
            objembedplay = false;
        }
    })

    //keras 제어
    $(".kerasbtnleft").children("#3000").click(function(){
        modelimgkeras.src = srcoverallimg("kerasembedding","3000",kerasobj);
        changebtncolor(".kerasbtnleft",'#3000');
    })
    $(".kerasbtnleft").children("#5000").click(function(){
        modelimgkeras.src = srcoverallimg("kerasembedding","5000",kerasobj);
        changebtncolor(".kerasbtnleft",'#5000');
    })
    $(".kerasbtnleft").children("#10000").click(function(){
        modelimgkeras.src = srcoverallimg("kerasembedding","10000",kerasobj);
        changebtncolor(".kerasbtnleft",'#10000');
    })
    $(".kerasbtnleft").children("#50000").click(function(){
        modelimgkeras.src = srcoverallimg("kerasembedding","50000",kerasobj);
        changebtncolor(".kerasbtnleft",'#50000');
    })
    $(".kerasbtnleft").children("#gt").click(function(){
        modelimgkeras.src = srcoverallimg("kerasembedding","gt",kerasobj);
        changebtncolor(".kerasbtnleft",'#gt');
    })

    $(".kerasbtnright").children("#3000").click(function(){
        gtimgkeras.src = srcoverallimg("kerasembedding","3000",kerasobj);
        changebtncolor(".kerasbtnright",'#3000');
    })
    $(".kerasbtnright").children("#5000").click(function(){
        gtimgkeras.src = srcoverallimg("kerasembedding","5000",kerasobj);
        changebtncolor(".kerasbtnright",'#5000');
    })
    $(".kerasbtnright").children("#10000").click(function(){
        gtimgkeras.src = srcoverallimg("kerasembedding","10000",kerasobj);
        changebtncolor(".kerasbtnright",'#10000');
    })
    $(".kerasbtnright").children("#50000").click(function(){
        gtimgkeras.src = srcoverallimg("kerasembedding","50000",kerasobj);
        changebtncolor(".kerasbtnright",'#50000');
    })
    $(".kerasbtnright").children("#gt").click(function(){
        gtimgkeras.src = srcoverallimg("kerasembedding","gt",kerasobj);
        changebtncolor(".kerasbtnright",'#gt');
    })
    $(".kerasinput").children('#selectobj').on("change", function(){
        kerasobj = $(this).find("option:selected").val();
    })

    // encode 제어
    $(".encodebtnleft").children('#3000').click(function(){
        if(encodeobj == 'cube'){
            //gtimgencode.src = "img/convencoding/gtcube3000.png";
            modelimgencode.src = "img/convencoding/cube3000.png";
            changebtncolor(".encodebtnleft",'#3000');
        }
        else if(encodeobj == 'greek'){
            //gtimgencode.src = "img/convencoding/gtgreek3000.png";
            modelimgencode.src = "img/convencoding/greek3000.png";
            changebtncolor(".encodebtnleft",'#3000');
        }
    });
    $(".encodebtnleft").children('#5000').click(function(){
        if(encodeobj == 'cube'){
            //gtimgencode.src = "img/convencoding/gtcube5000.png";
            modelimgencode.src = "img/convencoding/cube5000.png";
            changebtncolor(".encodebtnleft",'#5000');
        }
        else if(encodeobj == 'greek'){
            //gtimgencode.src = "img/convencoding/gtgreek5000.png";
            modelimgencode.src = "img/convencoding/greek5000.png";
            changebtncolor(".encodebtnleft",'#5000');
        }
    });
    $(".encodebtnleft").children('#10000').click(function(){
        if(encodeobj == 'cube'){
            //gtimgencode.src = "img/convencoding/gtcube10000.png";
            modelimgencode.src = "img/convencoding/cube10000.png";
            changebtncolor(".encodebtnleft",'#10000');
        }
        else if(encodeobj == 'greek'){
            //gtimgencode.src = "img/convencoding/gtgreek10000.png";
            modelimgencode.src = "img/convencoding/greek10000.png";
            changebtncolor(".encodebtnleft",'#10000');
        }
    });
    $(".encodebtnleft").children('#50000').click(function(){
        if(encodeobj == 'cube'){
            //gtimgencode.src = "img/convencoding/gtcube50000.png";
            modelimgencode.src = "img/convencoding/cube50000.png";
            changebtncolor(".encodebtnleft",'#50000');
        }
        else if(encodeobj == 'greek'){
            //gtimgencode.src = "img/convencoding/gtgreek50000.png";
            modelimgencode.src = "img/convencoding/greek50000.png";
            changebtncolor(".encodebtnleft",'#50000');
        }
    })
    $(".encodebtnleft").children('#gt').click(function(){
        if(encodeobj == 'cube'){
            //gtimgencode.src = "img/convencoding/gtcube50000.png";
            modelimgencode.src = "img/convencoding/gtcube.png";
            changebtncolor(".encodebtnleft",'#gt');
        }
        else if(encodeobj == 'greek'){
            //gtimgencode.src = "img/convencoding/gtgreek50000.png";
            modelimgencode.src = "img/convencoding/gtgreek.png";
            changebtncolor(".encodebtnleft",'#gt');
        }
    })

    $(".encodebtnright").children('#3000').click(function(){
        if(encodeobj == 'cube'){
            gtimgencode.src = "img/convencoding/cube3000.png";
            //modelimgencode.src = "img/convencoding/cube3000.png";
            changebtncolor(".encodebtnright",'#3000');
        }
        else if(encodeobj == 'greek'){
            gtimgencode.src = "img/convencoding/greek3000.png";
            //modelimgencode.src = "img/convencoding/greek3000.png";
            changebtncolor(".encodebtnright",'#3000');
        }
    });
    $(".encodebtnright").children('#5000').click(function(){
        if(encodeobj == 'cube'){
            gtimgencode.src = "img/convencoding/cube5000.png";
            //modelimgencode.src = "img/convencoding/cube5000.png";
            changebtncolor(".encodebtnright",'#5000');
        }
        else if(encodeobj == 'greek'){
            gtimgencode.src = "img/convencoding/greek5000.png";
            //modelimgencode.src = "img/convencoding/greek5000.png";
            changebtncolor(".encodebtnright",'#5000');
        }
    });
    $(".encodebtnright").children('#10000').click(function(){
        if(encodeobj == 'cube'){
            gtimgencode.src = "img/convencoding/cube10000.png";
            //modelimgencode.src = "img/convencoding/cube10000.png";
            changebtncolor(".encodebtnright",'#10000');
        }
        else if(encodeobj == 'greek'){
            gtimgencode.src = "img/convencoding/greek10000.png";
           // modelimgencode.src = "img/convencoding/greek10000.png";
           changebtncolor(".encodebtnright",'#10000');
        }
    });
    $(".encodebtnright").children('#50000').click(function(){
        if(encodeobj == 'cube'){
            gtimgencode.src = "img/convencoding/cube50000.png";
            //modelimgencode.src = "img/convencoding/cube50000.png";
            changebtncolor(".encodebtnright",'#50000');
        }
        else if(encodeobj == 'greek'){
            gtimgencode.src = "img/convencoding/greek50000.png";
            //modelimgencode.src = "img/convencoding/greek50000.png";
            changebtncolor(".encodebtnright",'#50000');
        }
    })
    $(".encodebtnright").children('#gt').click(function(){
        if(encodeobj == 'cube'){
            gtimgencode.src = "img/convencoding/gtcube.png";
           // modelimgencode.src = "img/convencoding/cube50000.png";
           changebtncolor(".encodebtnright",'#gt');
        }
        else if(encodeobj == 'greek'){
            gtimgencode.src = "img/convencoding/gtgreek.png";
            //modelimgencode.src = "img/convencoding/greek50000.png";
            changebtncolor(".encodebtnright",'#gt');
        }
    })



    $(".encodeinput").children('#selectobj').on("change", function(){
        encodeobj = $(this).find("option:selected").val();
    })
    $('.encodevideodiv').children('.video1').click(function(){
        if(encodeplay == false){
            $('.encodevideodiv').children('.video2').get(0).play();
            $('.encodevideodiv').children('.video1').get(0).play();
            encodeplay = true;
        }
        else if(encodeplay == true){
            $('.encodevideodiv').children('.video2').get(0).pause();
            $('.encodevideodiv').children('.video1').get(0).pause();
            encodeplay = false;
        }
    })
    $('.encodevideodiv').children('.video2').click(function(){
        if(encodeplay == false){
            $('.encodevideodiv').children('.video2').get(0).play();
            $('.encodevideodiv').children('.video1').get(0).play();
            encodeplay = true;
        }
        else if(encodeplay == true){
            $('.encodevideodiv').children('.video2').get(0).pause();
            $('.encodevideodiv').children('.video1').get(0).pause();
            encodeplay = false;
        }
    })
    $('.encodevideodiv2').children('.video1').click(function(){
        if(encodeplay2 == false){
            $('.encodevideodiv2').children('.video2').get(0).play();
            $('.encodevideodiv2').children('.video1').get(0).play();
            encodeplay2 = true;
        }
        else if(encodeplay2 == true){
            $('.encodevideodiv2').children('.video2').get(0).pause();
            $('.encodevideodiv2').children('.video1').get(0).pause();
            encodeplay2 = false;
        }
    })
    $('.encodevideodiv2').children('.video2').click(function(){
        if(encodeplay2 == false){
            $('.encodevideodiv2').children('.video2').get(0).play();
            $('.encodevideodiv2').children('.video1').get(0).play();
            encodeplay2 = true;
        }
        else if(encodeplay2 == true){
            $('.encodevideodiv2').children('.video2').get(0).pause();
            $('.encodevideodiv2').children('.video1').get(0).pause();
            encodeplay2 = false;
        }
    })
    const fun = spawn('python',['render.py',obj])
    fun.stdout.on('data',(result) =>{
        console.log(result);
    })
}