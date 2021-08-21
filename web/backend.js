window.onload = function(){
    var obgembedobj='cube';
    var encodeobj='cube';
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

    // objembed 제어
    $(".objembedbtnleft").children('#3000').click(function(){
        if(obgembedobj == 'cube'){
            //gtimg.src = "img/gtcube3000.png";
            modelimg.src = "img/cube3000.png";
            changebtncolor(".objembedbtnleft",'#3000');
        }
        else if(obgembedobj == 'greek'){
            //gtimg.src = "img/gtgreek3000.png";
            modelimg.src = "img/greek3000.png";
            changebtncolor(".objembedbtnleft",'#3000');
        }
    });
    $(".objembedbtnleft").children('#5000').click(function(){
        if(obgembedobj == 'cube'){
            //gtimg.src = "img/gtcube5000.png";
            modelimg.src = "img/cube5000.png";
            changebtncolor(".objembedbtnleft",'#5000');
        }
        else if(obgembedobj == 'greek'){
            //gtimg.src = "img/gtgreek5000.png";
            modelimg.src = "img/greek5000.png";
            changebtncolor(".objembedbtnleft",'#5000');
        }
    });
    $(".objembedbtnleft").children('#10000').click(function(){
        if(obgembedobj == 'cube'){
            //gtimg.src = "img/gtcube10000.png";
            modelimg.src = "img/cube10000.png";
            changebtncolor(".objembedbtnleft",'#10000');
        }
        else if(obgembedobj == 'greek'){
            //gtimg.src = "img/gtgreek10000.png";
            modelimg.src = "img/greek10000.png";
            changebtncolor(".objembedbtnleft",'#10000');
        }
    });
    $(".objembedbtnleft").children('#50000').click(function(){
        if(obgembedobj == 'cube'){
            //gtimg.src = "img/gtcube50000.png";
            modelimg.src = "img/cube50000.png";
            changebtncolor(".objembedbtnleft",'#50000');
        }
        else if(obgembedobj == 'greek'){
            //gtimg.src = "img/gtgreek50000.png";
            modelimg.src = "img/greek50000.png";
            changebtncolor(".objembedbtnleft",'#50000');
        }
    })
    $(".objembedbtnleft").children('#gt').click(function(){
        if(obgembedobj == 'cube'){
            //gtimg.src = "img/gtcube50000.png";
            modelimg.src = "img/gtcube.png";
            changebtncolor(".objembedbtnleft",'#gt');
        }
        else if(obgembedobj == 'greek'){
            //gtimg.src = "img/gtgreek50000.png";
            modelimg.src = "img/gtgreek.png";
            changebtncolor(".objembedbtnleft",'#gt');
        }
    })


    $(".objembedbtnright").children('#3000').click(function(){
        if(obgembedobj == 'cube'){
            gtimg.src = "img/cube3000.png";
            //modelimg.src = "img/cube003000.png";
            changebtncolor(".objembedbtnright",'#3000');
        }
        else if(obgembedobj == 'greek'){
            gtimg.src = "img/greek3000.png";
            //modelimg.src = "img/greek003000.png";
            changebtncolor(".objembedbtnright",'#3000');
        }
    });
    $(".objembedbtnright").children('#5000').click(function(){
        if(obgembedobj == 'cube'){
            gtimg.src = "img/cube5000.png";
            //modelimg.src = "img/cube005000.png";
            changebtncolor(".objembedbtnright",'#5000');
        }
        else if(obgembedobj == 'greek'){
            gtimg.src = "img/greek5000.png";
            //modelimg.src = "img/greek005000.png";
            changebtncolor(".objembedbtnright",'#5000');
        }
    });
    $(".objembedbtnright").children('#10000').click(function(){
        if(obgembedobj == 'cube'){
            gtimg.src = "img/cube10000.png";
            //modelimg.src = "img/cube010000.png";
            changebtncolor(".objembedbtnright",'#10000');
        }
        else if(obgembedobj == 'greek'){
            gtimg.src = "img/greek10000.png";
            //modelimg.src = "img/greek010000.png";
            changebtncolor(".objembedbtnright",'#10000');
        }
    });
    $(".objembedbtnright").children('#50000').click(function(){
        if(obgembedobj == 'cube'){
            gtimg.src = "img/cube50000.png";
            //modelimg.src = "img/cube050000.png";
            changebtncolor(".objembedbtnright",'#50000');
        }
        else if(obgembedobj == 'greek'){
            gtimg.src = "img/greek50000.png";
            //modelimg.src = "img/greek050000.png";
            changebtncolor(".objembedbtnright",'#50000');
        }
    })
    $(".objembedbtnright").children('#gt').click(function(){
        if(obgembedobj == 'cube'){
            gtimg.src = "img/gtcube.png";
            //modelimg.src = "img/gtcube.png";
            changebtncolor(".objembedbtnright",'#gt');
        }
        else if(obgembedobj == 'greek'){
            gtimg.src = "img/gtgreek.png";
           // modelimg.src = "img/gtgreek.png";
           changebtncolor(".objembedbtnright",'#gt');
        }
    })
    
    $(".objembedinput").children('#selectobj').on("change", function(){
        obgembedobj = $(this).find("option:selected").val();
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